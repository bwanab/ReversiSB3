#!/usr/bin/env python3
"""
Fine-tune a model on its own MCTS self-play (selfplay_mcts.py), mixed with a replay of Edax labels
(label_positions.py --every-move) so it doesn't drift where the self-play data is thin. The step
toward AlphaZero's loop: the network learns from its own search (CLAUDE.md step 28).

Targets per position:
  policy  MCTS rows: the visit distribution; Edax rows: the graded target softmax(move_score / T)
  value   MCTS rows: (1 - z) * root_value + z * tanh(outcome / scale) (--z-weight z: 0 = trust the
          search's evaluation, 1 = the game's result, as AlphaZero); Edax rows: tanh(score / scale)
All 8 board symmetries are used for augmentation. Validation: Edax-labeled positions (policy and
1-ply value regret in discs, as edax_train.py) and held-out MCTS positions (policy cross-entropy
against the visits, value MSE). Saves a model after every epoch.

Usage:
  python mcts_train.py --mcts selfplay_r256x12_st1.npz --replay labels_d*_all.npz --replay-n 1000000 \\
      --base r256x12_mid1 --model r256x12_st1 --epochs 2 -lr 2e-5
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO

from edax_train import load_labels, evaluate, heads
from util.board_features import PERMS, legal_moves as legal_planes
from util.reversi import build_reversi
from util.util import get_device


def legal_mask(boards, chunk=65536):
    out = []
    for i in range(0, len(boards), chunk):
        b = torch.as_tensor(boards[i:i + chunk].reshape(-1, 8, 8))
        out.append(legal_planes((b == 1).float(), (b == -1).float()).reshape(-1, 64).bool().numpy())
    return np.concatenate(out)


def graded(move_scores, temperature):
    legal = ~np.isnan(move_scores)
    x = np.where(legal, np.nan_to_num(move_scores) / temperature, -np.inf)
    x = np.exp(x - x.max(axis=1, keepdims=True))
    return (x / x.sum(axis=1, keepdims=True)).astype(np.float32), legal


@torch.no_grad()
def mcts_metrics(policy, val, device, batch=2048):
    policy.set_training_mode(False)
    ce, mse = [], []
    for i in range(0, len(val["boards"]), batch):
        logits, v = heads(policy, val["boards"][i:i + batch], device)
        legal = torch.as_tensor(val["legal"][i:i + batch], device=device)
        logp = torch.log_softmax(logits.masked_fill(~legal, -1e9), dim=1)
        pi = torch.as_tensor(val["pi"][i:i + batch], device=device)
        ce.append(-(pi * logp).sum(dim=1).cpu().numpy())
        mse.append(((v - torch.as_tensor(val["v"][i:i + batch], device=device)) ** 2).cpu().numpy())
    policy.set_training_mode(True)
    return {"mcts_policy_ce": float(np.concatenate(ce).mean()), "mcts_value_mse": float(np.concatenate(mse).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mcts", nargs="+", required=True, help="selfplay_mcts.py output file(s)")
    parser.add_argument("--replay", nargs="+", required=True, help="Edax every-move label files")
    parser.add_argument("--replay-n", type=int, default=1000000, help="Edax positions sampled for replay")
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--z-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--value-scale", type=float, default=16.0)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("-lr", "--learning-rate", type=float, default=2e-5)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # Edax replay (and its validation split, with move scores for the regret metrics)
    edax_train_set, edax_val = load_labels(args.replay, args.val_frac, args.seed)
    take = rng.permutation(len(edax_train_set["boards"]))[:args.replay_n]
    e_boards = edax_train_set["boards"][take]
    e_pi, e_legal = graded(edax_train_set["move_scores"][take], args.temperature)
    e_v = np.tanh(edax_train_set["score"][take] / args.value_scale).astype(np.float32)
    edax_val = {k: (None if v is None else v[:20000]) for k, v in edax_val.items()}

    # MCTS self-play
    files = [np.load(p) for p in args.mcts]
    m_boards = np.concatenate([f["boards"] for f in files]).astype(np.int8)
    m_pi = np.concatenate([f["visits"] for f in files]).astype(np.float32)
    q = np.concatenate([f["root_value"] for f in files])
    z = np.tanh(np.concatenate([f["outcome"] for f in files]) / args.value_scale)
    m_v = ((1 - args.z_weight) * q + args.z_weight * z).astype(np.float32)
    m_legal = legal_mask(m_boards)
    order = rng.permutation(len(m_boards))
    n_val = int(len(order) * args.val_frac)
    mval = {"boards": m_boards[order[:n_val]], "pi": m_pi[order[:n_val]], "v": m_v[order[:n_val]],
            "legal": m_legal[order[:n_val]]}
    tr = order[n_val:]

    boards = np.concatenate([m_boards[tr], e_boards])
    pi = np.concatenate([m_pi[tr], e_pi])
    legal = np.concatenate([m_legal[tr], e_legal])
    v = np.concatenate([m_v[tr], e_v])
    print(f"{len(tr):,} MCTS + {len(e_boards):,} Edax replay training positions; "
          f"{n_val:,} MCTS / {len(edax_val['boards']):,} Edax validation; value target z-weight {args.z_weight}")

    device = get_device() if args.device == "auto" else args.device
    model = MaskablePPO.load(f"models/{args.base}_CNN_test", env=build_reversi("Random"), device=device)
    policy = model.policy
    device = policy.device
    report = lambda: {**evaluate(policy, edax_val, device, args.value_scale), **mcts_metrics(policy, mval, device)}
    print("start:", report(), flush=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    steps = args.epochs * (len(boards) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    for epoch in range(args.epochs):
        policy.set_training_mode(True)
        perm = rng.permutation(len(boards))
        total, n = 0.0, 0
        for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            ix = perm[i:i + args.batch_size]
            sym = PERMS[rng.integers(0, 8, size=len(ix))]                 # new[j] = old[sym[j]]
            b = np.take_along_axis(boards[ix], sym, axis=1)
            p = torch.as_tensor(np.take_along_axis(pi[ix], sym, axis=1), device=device)
            lg = torch.as_tensor(np.take_along_axis(legal[ix], sym, axis=1), device=device)
            logits, val = heads(policy, b, device)
            logp = torch.log_softmax(logits.masked_fill(~lg, -1e9), dim=1)
            loss = -(p * logp).sum(dim=1).mean() + args.value_coef * F.mse_loss(
                val, torch.as_tensor(v[ix], device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total, n = total + loss.item(), n + 1
        print(f"epoch {epoch + 1}: train loss {total / n:.4f}  val {report()}", flush=True)
        model.save(f"models/{args.model}_epoch{epoch + 1:02d}_CNN_test")
    model.save(f"models/{args.model}_CNN_test")
    print(f"saved models/{args.model}_CNN_test.zip")


if __name__ == "__main__":
    main()
