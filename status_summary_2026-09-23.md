# ReversiSB3 — Status Summary (2026-09-23)

## Modified Files: Assessment

All 12 modified files are valid and consistent with the recent work stream (Edax integration + curriculum training). None should be discarded.

### Core feature work — commit-worthy

- **[sb-train.py](sb-train.py)** — Adds the new `edax-curriculum` training mode via `train_edax_curriculum()`. Supports `--edax-depths 6,7,8` and `--edax-ratios 0.3,0.3,0.3` with optional random-opponent mix and configurable `--refresh-interval`. Also renames short flags: `-d/--rai-depth` and `-x/--edax-depth`.
  - Nit: dead commented-out `multiprocessing.set_start_method('spawn')` block at the top ([sb-train.py:1-7](sb-train.py#L1-L7)). Delete before committing.

- **[util/opponents.py](util/opponents.py)** — Refactors `EdaxOpponent` from direct C bindings to a Unix-socket client (`EdaxClient`). Necessary because SB3's parallel env forking corrupted the shared C-library state. The server side is [edax_server.py](edax_server.py) (untracked); see [EDAX_SERVER_USAGE.md](EDAX_SERVER_USAGE.md).

- **[util/edax_engine.py](util/edax_engine.py)** — Docstring updates only; notes that it's now used by `edax_server.py`.

- **[bc_train.py](bc_train.py)** — Handles the new BC dataset metadata format (`edax_depths`, `merged_from`, `merged_source`, `total_moves`) for combined WThor + Edax datasets. Backward-compatible with the older RAI-only format.

- **[sb-play.py](sb-play.py)** — Adds a color-balanced Model-vs-Model playoff mode (half the games each color) and reports "Model advantage". Matches how [edax_bc_playoff.sh](edax_bc_playoff.sh) is being used to compare BC epochs.

- **[test_edax_opponent.py](test_edax_opponent.py)** — Rewritten to use `gym.make("ReversiCNN-v0", opponent="Edax", depth=4)` instead of manual env construction. Matches training-time usage.

- **[web_app_spec.txt](web_app_spec.txt)** — Adds clarifying documentation of the API (return shapes, param formats). Documentation-only.

### Housekeeping — technically valid, low value

- **[sanity1.py](sanity1.py) – [sanity5.py](sanity5.py)** — All switch `ReversiEnvCNN.build_reversi()` → module-level `build_reversi()` (the class method was moved). [sanity3.py](sanity3.py) also uncomments some `render()` calls for debug output. Harmless.

### Recommended commit strategy

Split into two commits (optional):
1. **`Add edax-curriculum training mode and client/server Edax opponent`** — sb-train.py (after removing dead comment block), util/opponents.py, util/edax_engine.py, bc_train.py, sb-play.py, test_edax_opponent.py, web_app_spec.txt
2. **`Sanity scripts: use module-level build_reversi()`** — sanity1-5.py

Or a single squash if you prefer.

Note: there are many untracked files (`test_*.py`, `benchmark_*.py`, various `.pkl`, `.csv`, `.sh`, `.md`). Most are scratch/experimental. Worth a separate pass to triage — some (e.g. [edax_server.py](edax_server.py), [generate_edax_bc_dataset.py](generate_edax_bc_dataset.py), [util/edax_client.py](util/edax_client.py), [util/edax_gtp.py](util/edax_gtp.py), [EDAX_SERVER_USAGE.md](EDAX_SERVER_USAGE.md)) are load-bearing for the current workflow and should be committed. Others are likely one-off tests to `.gitignore` or delete.

---

## Current Project State

### Active model

`models/edax_bc_pretrained_CNN_test.zip` — Behavioral Cloning pretrained on combined WThor games + Edax-7/8/9 games (epoch 18 selected as best), then approximately 11M+ RL timesteps alternating `edax-curriculum` and `selfplay` / `mixed` modes with LR decayed from 1e-5 down to 3e-7.

### Training progression (from [exax_pretrain_steps.doc](exax_pretrain_steps.doc))

The `.doc` file contains 25 numbered training blocks. Each records the exact `sb-train.py` command and the win-rate results against Edax at various depths. Approximate pattern:

- **Steps 1–3**: Build WThor BC dataset; add Edax-7/8/9 games; BC-pretrain 20 epochs.
- **Steps 4–8**: Curriculum-train against Edax-1 through Edax-5 with 500k–1M timestep blocks, alternating with selfplay.
- **Steps 9–16**: Push into Edax-6, then Edax-7/8/9. Ratios rebalanced repeatedly.
- **Steps 17–25**: LR aggressively decayed (1e-6 → 5e-7 → 3e-7). Mixed self-play with occasional RAI. Curriculum widened to include Edax-10.

### Latest evaluated win rates (step 25 of 25)

| Opponent | Win % |
|----------|-------|
| Edax-3 | 89% |
| Edax-4 | 94% |
| Edax-5 | 97% |
| Edax-6 | 97% |
| Edax-7 | 69% |
| **Edax-8** | **0%** ← wall |
| Edax-9 | 23% |
| **Edax-10** | **0%** |

### Goals vs. reality

- Original CLAUDE.md targets were "beat RAI-4 in majority of games." Long since exceeded (RAI-1: 87%, RAI-2: 86% back when [session_notes.md](session_notes.md) was written, and Edax is significantly stronger than RAI at every depth).
- Current goal (implicit): push into higher Edax depths. Progress is real up through Edax-6, but stalled at Edax-8 and Edax-10.

### The discontinuity

The most interesting anomaly: **97% at Edax-6 → 0% at Edax-8 → 23% at Edax-9 → 0% at Edax-10.**

This is not a smooth curve. If the model were simply "outclassed above depth 6," you'd expect a monotonic decline. Instead there's a specific failure at Edax-8 and Edax-10 with a partial rebound at Edax-9.

Possible explanations:
1. **Even-depth pathology.** Minimax-based engines can play qualitatively differently at even vs. odd depths (parity of the search horizon relative to end-of-game). The model may have implicitly learned to counter odd-depth play through the earlier curriculum, but not even-depth.
2. **Specific tactical threshold.** Edax-8+ may unlock a particular pattern (e.g., forced sacrifice, endgame parity trap) that Edax-6 never plays. The model has no defense because it never saw it during training.
3. **Deterministic loss loop.** If the deterministic-play evaluation always produces the same game, a 0% is one bad game repeating. Worth confirming whether the eval was `-d` (deterministic) or non-deterministic — the [edax_bc_playoff.sh](edax_bc_playoff.sh) scripts use `-d`.

---

## Suggested Next Steps

Ordered from cheapest / most informative first:

### 1. Diagnose the Edax-8 wall before more training

Play 5–10 Edax-8 games with `-v` (verbose) and non-deterministic mode. Look at:
- Does the model always lose in the same way?
- Is Edax-8 playing a specific opening/tactic that Edax-6 doesn't?
- At what move does the position visibly go wrong?

If it's a deterministic-eval artifact, the answer is very different than if it's a genuine strategic gap. This takes ~30 minutes and directly informs whether more training would help.

### 2. Verify the eval methodology

Confirm the reported win rates are non-deterministic. In [session_notes.md](session_notes.md) the earlier model showed 100% deterministic (same game replayed) vs. ~6% non-deterministic. Same trap may explain some of the 0% results now.

### 3. If it's a strategic gap: fresh BC boost from Edax-8/9/10

The current BC dataset was built from Edax-7/8/9 (per exax_pretrain_steps.doc step 2). But the RL fine-tuning may have drifted the policy away from that knowledge. Options:
- Generate a small targeted BC dataset (~2000 games) from Edax-8 and Edax-10 specifically.
- Joint-train (policy + value) for a few epochs to reinject that knowledge.
- Then resume the curriculum with a slightly higher LR (1e-6 instead of 3e-7) so it can actually absorb it.

### 4. If it's diminishing returns: pivot to generalization

The [session_notes.md](session_notes.md) file flagged that beating RAI didn't translate to beating the Piccolo iPhone app — the model was RAI-specialized. Same risk now exists for Edax. Worth checking:
- Does the current model still beat Piccolo levels 3–4?
- Does it beat other Reversi engines (if you can find them)?

If it does — declare victory and write up. If it doesn't — the "specialization" problem is now the interesting research question, and more Edax training may make it worse.

### 5. Not recommended

- **More curriculum training at the same LR (3e-7) without diagnosis.** The last 3 steps (23, 24, 25) have shown flat or oscillating results at Edax-7+; more of the same is unlikely to break through.
- **Restarting from BC.** The BC-pretrained model is your best asset. Don't throw it away.

---

## Files worth reading yourself

- [exax_pretrain_steps.doc](exax_pretrain_steps.doc) — Authoritative training-run log, 25 numbered steps with commands and results.
- [session_notes.md](session_notes.md) — Earlier context: RAI-specialization problem, Piccolo iPhone app comparison.
- [edax_train_results.csv](edax_train_results.csv) — Per-block eval numbers, matches the .doc file.
- [edax_bc_pretrained_training.csv](edax_bc_pretrained_training.csv) — BC training curves (train/val loss, random_wins, rai1_wins per epoch).
- [EDAX_SERVER_USAGE.md](EDAX_SERVER_USAGE.md) — Architecture of the Edax client/server (needed to run any Edax-based training).
