#!/usr/bin/env python3
"""
Fine-tune a model on Edax labels from label_positions.py (supervised, like BC).

Targets:
  policy  'graded': soft target over legal moves, p(m) ~ exp(move_score(m) / T), needs
          --every-move labels; 'best': one-hot on Edax's best move; 'score': regress the
          policy logits onto move_score / T over legal moves, both centered per position
          (only differences between moves matter), so softmax(logits) -> exp(score / T) and
          every move's score is fit equally (needs --every-move labels)
  value   tanh(edax_score / value_scale), the value head's usual [-1, 1] range
All 8 board symmetries are used for augmentation (boards and move scores permuted together).

Validation (unaugmented), in discs where it says so:
  policy regret  max move score - score of the policy's top move (0 = always Edax-best)
  value regret   same for the move a 1-ply value-head search picks (needs --every-move labels)
  top-1          policy's top move == Edax's best move
  value MSE      against the tanh-scaled target
With --epochs 0 this just reports the metrics for the starting model.

Usage:
  python edax_train.py --labels labels_d12_100k_all.npz --base resnet128x8_bc20cos_bconly \\
      --model edaxft_policy --train both --policy-target graded --epochs 3
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO

from util.reversi import build_reversi
from util.board_features import PERMS
from util.search import play_move, legal_moves
from util.util import get_device


def load_labels(paths, val_frac, seed):
    """Load and concatenate label files, dropping repeated positions (first one kept)."""
    files = [np.load(p) for p in paths]
    has_ms = all("move_scores" in L for L in files)
    boards = np.concatenate([L["boards"] for L in files]).astype(np.int8)
    _, first = np.unique(boards, axis=0, return_index=True)
    keep = np.sort(first)
    data = dict(boards=boards[keep],
                best=np.concatenate([L["best_move"] for L in files]).astype(np.int64)[keep],
                score=np.concatenate([L["score"] for L in files]).astype(np.float32)[keep],
                move_scores=np.concatenate([L["move_scores"] for L in files]).astype(np.float32)[keep]
                if has_ms else None)
    if len(keep) < len(boards):
        print(f"dropped {len(boards) - len(keep):,} repeated positions across label files")
    boards = data["boards"]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(boards))
    n_val = int(len(boards) * val_frac)
    split = lambda ix: {k: (None if v is None else v[ix]) for k, v in data.items()}
    return split(idx[n_val:]), split(idx[:n_val])


def augment(boards, best, move_scores, rng):
    """Apply a random symmetry per example: new[j] = old[PERMS[k][j]]; moves via the inverse."""
    k = rng.integers(0, 8, size=len(boards))
    perms = PERMS[k]                                       # (B, 64)
    boards = np.take_along_axis(boards, perms, axis=1)
    inv = np.argsort(perms, axis=1)
    best = inv[np.arange(len(best)), best]
    if move_scores is not None:
        move_scores = np.take_along_axis(move_scores, perms, axis=1)
    return boards, best, move_scores


def heads(policy, boards, device):
    obs = torch.as_tensor(boards.reshape(-1, 1, 8, 8), dtype=torch.float32, device=device)
    features = policy.extract_features(obs)
    latent_pi, latent_vf = policy.mlp_extractor(features)
    return policy.action_net(latent_pi), policy.value_net(latent_vf).squeeze(-1)


def policy_loss(logits, best, move_scores, target, temperature, device, value_scale):
    if target == "best":
        return F.cross_entropy(logits, torch.as_tensor(best, device=device))
    ms = torch.as_tensor(move_scores, device=device)
    legal = ~torch.isnan(ms)
    if target == "score":
        n_legal = legal.sum(dim=1, keepdim=True).clamp(min=1)
        t = torch.nan_to_num(ms) / temperature
        lg = logits.masked_fill(~legal, 0.0)
        t_c = t - t.masked_fill(~legal, 0.0).sum(dim=1, keepdim=True) / n_legal
        l_c = lg - lg.sum(dim=1, keepdim=True) / n_legal
        return ((l_c - t_c) ** 2)[legal].mean()
    target_logits = torch.where(legal, torch.nan_to_num(ms) / temperature, torch.full_like(ms, -1e9))
    target_p = torch.softmax(target_logits, dim=1)
    logp = torch.log_softmax(logits.masked_fill(~legal, -1e9), dim=1)
    return -(target_p * logp).sum(dim=1).mean()


@torch.no_grad()
def evaluate(policy, val, device, value_scale, batch=2048):
    policy.set_training_mode(False)
    boards, best, score, ms = val["boards"], val["best"], val["score"], val["move_scores"]
    top1, pol_regret, vmse = [], [], []
    for i in range(0, len(boards), batch):
        logits, v = heads(policy, boards[i:i + batch], device)
        if ms is not None:
            logits = logits.masked_fill(torch.as_tensor(np.isnan(ms[i:i + batch]), device=device), -1e9)
        pick = logits.argmax(dim=1).cpu().numpy()
        top1.append(pick == best[i:i + batch])
        if ms is not None:
            m = ms[i:i + batch]
            pol_regret.append(np.nanmax(m, axis=1) - m[np.arange(len(pick)), pick])
        vmse.append(((v.cpu().numpy() - np.tanh(score[i:i + batch] / value_scale)) ** 2))
    out = {"top1": float(np.concatenate(top1).mean()), "value_mse": float(np.concatenate(vmse).mean())}
    if ms is not None:
        out["policy_regret"] = float(np.concatenate(pol_regret).mean())
        # 1-ply value search: pick the move whose resulting position the value head likes least
        # for the opponent (children are from the opponent's view, so score = -V(child))
        regrets = []
        sub = range(min(len(boards), 5000))
        for i in sub:
            moves = legal_moves(boards[i])
            children = np.stack([play_move(boards[i], m) for m in moves])
            _, vc = heads(policy, children, device)
            pick = moves[int(torch.argmax(-vc))]
            regrets.append(np.nanmax(ms[i]) - ms[i][pick])
        out["value_regret"] = float(np.mean(regrets))
    policy.set_training_mode(True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True, nargs="+", help="one or more label .npz files")
    parser.add_argument("--base", required=True, help="starting model name under models/ (no _CNN_test.zip)")
    parser.add_argument("--model", help="output model name (saved as models/{model}_CNN_test.zip)")
    parser.add_argument("--train", choices=["policy", "value", "both"], default="both")
    parser.add_argument("--policy-target", choices=["graded", "best", "score"], default="graded")
    parser.add_argument("--temperature", type=float, default=2.0, help="graded target: discs per e-fold")
    parser.add_argument("--value-scale", type=float, default=16.0, help="value target = tanh(score / scale)")
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--freeze-trunk", action="store_true", help="train only the head(s), not the trunk")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("-lr", "--learning-rate", type=float, default=3e-5)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--device", default="auto", help="auto = cuda, else mps, else cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train, val = load_labels(args.labels, args.val_frac, args.seed)
    if args.policy_target in ("graded", "score") and train["move_scores"] is None and args.train != "value":
        parser.error(f"--policy-target {args.policy_target} needs labels made with --every-move")
    env = build_reversi("Random")
    device = get_device() if args.device == "auto" else args.device
    model = MaskablePPO.load(f"models/{args.base}_CNN_test", env=env, device=device)
    policy = model.policy
    device = policy.device
    print(f"{len(train['boards']):,} train / {len(val['boards']):,} val positions; base {args.base}")
    print("start:", evaluate(policy, val, device, args.value_scale))

    if args.epochs == 0:
        return
    if args.freeze_trunk:
        for p in policy.features_extractor.parameters():
            p.requires_grad_(False)
        policy.features_extractor.eval()  # keep BatchNorm statistics fixed too
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    steps = args.epochs * (len(train["boards"]) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        policy.set_training_mode(True)
        if args.freeze_trunk:
            policy.features_extractor.eval()
        # a head that isn't being trained must not update its BatchNorm statistics either
        heads_ = getattr(policy.mlp_extractor, "policy_head", None), getattr(policy.mlp_extractor, "value_head", None)
        if args.train == "value" and heads_[0] is not None:
            heads_[0].eval()
        if args.train == "policy" and heads_[1] is not None:
            heads_[1].eval()
        order = rng.permutation(len(train["boards"]))
        total, n = 0.0, 0
        for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
            ix = order[i:i + args.batch_size]
            ms = None if train["move_scores"] is None else train["move_scores"][ix]
            boards, best, ms = augment(train["boards"][ix], train["best"][ix], ms, rng)
            logits, v = heads(policy, boards, device)
            loss = 0.0
            if args.train in ("policy", "both"):
                loss = loss + policy_loss(logits, best, ms, args.policy_target, args.temperature, device,
                                          args.value_scale)
            if args.train in ("value", "both"):
                target = torch.as_tensor(np.tanh(train["score"][ix] / args.value_scale), device=device)
                loss = loss + args.value_coef * F.mse_loss(v, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total, n = total + loss.item(), n + 1
        print(f"epoch {epoch + 1}: train loss {total / n:.4f}  val {evaluate(policy, val, device, args.value_scale)}",
              flush=True)

    if args.model:
        if args.freeze_trunk:
            for p in policy.features_extractor.parameters():
                p.requires_grad_(True)
        model.save(f"models/{args.model}_CNN_test")
        print(f"saved models/{args.model}_CNN_test.zip")


if __name__ == "__main__":
    main()
