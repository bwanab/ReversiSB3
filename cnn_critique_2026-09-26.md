# ReversiCNN critique (2026-09-26)

A review of `util/reversi_cnn.py` and the policy built around it, measured against the project goal
of beating progressively deeper Edax searches (the previous best model beats Edax-6 ~97% but gets
0% at Edax-8). Based on the source, the policy SB3 actually builds (printed with parameter counts),
the observation encoding in `util/reversi.py`, and `bc_train.py`.

## What the network actually is

`ReversiCNN` is only the trunk. `CnnPolicy` adds SB3's default heads because `net_arch` is never
set in `get_model()` (`util/util.py`). The full model:

```
board (1×8×8, values -1/0/1)
 → 5× conv3x3 + ReLU (64, 64, 128, 128, 256)     ~0.55M params
 → flatten 16384 → Linear 256 + ReLU              ~4.19M params  (87% of the model)
 → policy: 256 → 64 → 64 (Tanh) → 64 logits
 → value:  256 → 64 → 64 (Tanh) → 1
total ≈ 4.79M parameters
```

Parameter breakdown:

| Layer | Params |
|---|---|
| conv 1→64 | 640 |
| conv 64→64 | 36,928 |
| conv 64→128 | 73,856 |
| conv 128→128 | 147,584 |
| conv 128→256 | 295,168 |
| **linear 16384→256** | **4,194,560** |
| policy MLP (256→64→64) + action_net (64→64) | 24,768 |
| value MLP (256→64→64) + value_net (64→1) | 20,673 |

## Problems, roughly by impact

### 1. Most of the model is one fully-connected layer

Of 4.79M parameters, 4.19M sit in the 16384→256 layer. The conv stack computes features for each
square, then the network throws away where they are and relearns position through a dense matrix.

- That layer is where overfitting risk lives. BC val accuracy stalls at ~52%, and RL at learning
  rates like 3e-7 is fine-tuning mostly this one layer.
- The 5 conv layers alone see only an 11×11 window around each square (±5). That doesn't cover the
  longest lines from a corner (7 squares), so the FC layer carries all long-range reasoning. In
  Othello that means flanking lines, diagonals, and corner/edge interactions.

### 2. The policy head forces a spatial answer through a narrow, poorly suited path

A 64-way move choice goes 8×8×256 → 256 → 64 (tanh) → 64 → 64 logits. Move choice is naturally
spatial: "how good is playing on this square" is best read off each square's own features with a
1×1 conv giving one logit per square. AlphaZero-style Othello and Go nets do this.

The tanh MLP is SB3's default for vector observations (`pi=[64, 64], vf=[64, 64]`), not a
deliberate choice. The value head has the same bottleneck, although for a scalar that matters less.

### 3. The input encoding hides what matters most

The observation is a single channel with ±1 for stones and 0 for empty.

- **"Empty" can't be detected linearly.** Empty is 0, so a single filter weight gives it zero
  response. The net has to build it from two ReLUs, which wastes early capacity. Empty squares next
  to opponent stones are exactly what define a legal move.
- **No mobility.** The network has to derive legal moves itself, but they are free to compute: the
  env already has them for action masking (`get_valid`). Mobility (your legal moves vs. the
  opponent's) and frontier squares are the core of Othello strategy, and Edax's evaluation is built
  around them.

The standard encoding is separate binary planes:

1. own stones
2. opponent stones
3. empty
4. own legal moves
5. opponent legal moves
6. (optional) a constant plane for parity or number of empties

This is a cheap change with a large expected payoff.

### 4. Othello's 8-fold board symmetry isn't used

The board looks the same under 4 rotations and 4 reflections. I found no rotation or flip
augmentation anywhere in the repo (no `rot90`, `flip`, or `fliplr` in the BC or training code).

Augmenting the 3.45M-sample BC set would give ~27M effectively distinct positions for free. It would
also directly counter the FC layer's tendency to memorize specific squares. RL rollouts can also
apply a random symmetry per position.

### 5. It's plain and shallow for the target strength

There are no residual connections and no normalization. Strong learned Othello evaluators use
roughly 6–10 residual blocks of 64–128 channels with BatchNorm, with no FC trunk. Five plain convs
are trainable, but the effective depth for multi-step tactics is small. Beating Edax-8 needs a
network whose single forward pass approximates what an 8-ply search finds.

### 6. Policy and value share a trunk under PPO

`CnnPolicy` shares the features extractor between the actor and the critic by default. That is fine
for BC, which trains both heads jointly. Under PPO, value-loss gradients (vf_coef 0.5) and policy
gradients pull on the same 4M-parameter FC layer.

That may contribute to the swings seen in the training log, where self-play blocks wrecked Edax
win rates (e.g. step 4 → 5: Edax-3 75% → 1%). It's worth trying `share_features_extractor=False`,
or keeping a shared trunk with a much smaller FC layer.

## Suggested architecture for the retrain

- **Input:** 4–6 planes (own, opponent, empty, own legal moves, opponent legal moves, maybe parity).
- **Trunk:** a conv stem, then ~6–8 residual blocks at 64–128 channels with BatchNorm. No flatten
  into the trunk.
- **Policy head:** 1×1 conv → 1 channel → flatten → 64 logits.
- **Value head:** 1×1 conv → small FC → tanh scalar.
- **Wiring:** implement it as a custom `MaskableActorCriticPolicy` (or override
  `_build_mlp_extractor`) so the policy head stays spatial. A plain `BaseFeaturesExtractor` can't do
  that, because SB3 flattens its output before the heads.
- **Data:** 8-fold symmetry augmentation in BC, and optionally random symmetry transforms in RL.
- **Size:** a network like this lands around 1–2M parameters. That's fewer than now, but they'd be
  spent where they help.

### Effort and order

- The residual trunk plus spatial heads is roughly a day of work. It also needs a small change to
  `bc_train.py`, which currently calls `policy.extract_features` → `policy.mlp_extractor` →
  `action_net`/`value_net` directly.
- The input planes and symmetry augmentation are cheaper and could be tested first on the current
  architecture, using a quick BC run (1–2 epochs, ~2.3 min each on MPS) and comparing validation
  accuracy against the 2-epoch baseline (~52–53%).

## A caveat about the Edax-8 wall

A better network will raise the ceiling, but a policy that picks moves in one forward pass is
playing against an 8-ply alpha-beta search. The Edax-8 0% vs Edax-9 23% pattern still looks like a
specific tactical blind spot that no single change here obviously explains (see the diagnosis plan
in `status_summary_2026-09-23.md`).

The biggest single lever for beating deep Edax is probably adding search at play time: shallow
alpha-beta or MCTS using the value head. That only works if the value head is good, which is
another argument for points 2, 3 and 6.
