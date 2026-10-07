# Can a blank-slate network learn to play strong Othello by playing? (discussion, 2026-10-07)

A conversation held while the probability-pruning search test (CLAUDE.md, "Train of reasoning",
step 25) was running. It steps back from the step-by-step work to ask what the project has shown
about the original goal. The questions are quoted as asked; the answers are expanded from the
discussion, with references to the measurements in CLAUDE.md.

Contents:
1. Can it be done?
2. What is PPO good for?
3. Have we reinvented AlphaZero?
4. Takeaways and possible next experiments

---

## 1. Can it be done?

> In the README from the inception of the project I'd listed some goals that included % wins
> against the various targets. In fact, those were more accurately proximate goals where the
> ultimate goal I had was to build a strong Othello player using reinforcement learning in the
> sense of training a network by playing games until it learned how to win. I'm happy to have
> deviated from that goal to the extent that I've learned a lot about how a real life strategy
> game player is constructed, but I'm left with the question: can it be done? Can a blank slate
> neural net purely trained by game playing practically ever learn to win against a strong
> opponent? I've seen first hand that it can be trained to win against weak opponents in that even
> the old CNN model was able to beat me consistently. My thought is that theoretically this should
> be possible. If so, what practical considerations make that theory unachievable? If not, why not?

### Short answer

Yes. It has been done several times, including for games in Othello's family. What failed in this
project was one particular version of "learning by playing" (PPO on win/loss rewards), not the idea
itself.

### Existence proofs

- **TD-Gammon (Tesauro, 1992):** a neural network trained only by playing backgammon against
  itself, using temporal-difference learning, reached close to world-champion level.
- **AlphaGo Zero / AlphaZero (DeepMind, 2017):** started from random weights and the rules alone
  and surpassed every previous program in Go, chess and shogi.
- **Othello engines:** Logistello (which beat world champion Takeshi Murakami 6-0 in 1997), Edax,
  and (as far as I know) Egaroucid fit their evaluation functions to positions from games and
  search, not to human games or hand-tuned weights. Their pattern-based evaluations are not neural
  networks of our kind, but the principle (learn evaluation from the program's own play, labeled by
  outcome or search) is the same.

So a blank slate learning from the rules and play is not ruled out in principle, or in practice.

### What separates AlphaZero from what we tried

AlphaZero does not learn from wins and losses alone. At every move of every self-play game it runs
a search (Monte Carlo Tree Search, MCTS) guided by the network:

- The search's move preferences (how often it visited each move) become the network's **policy
  target**.
- The game's final result becomes the **value target**.

The search acts as a *policy-improvement operator*: even a weak network plus search plays better
than the network alone, so the network always trains toward something slightly better than itself.
The teacher is the network's own search. As the network improves, the search improves, and the
targets improve with it.

PPO with outcome rewards has no such teacher. We ran into each of its weaknesses (CLAUDE.md, steps
2-5 and the "Top-move finding"):

1. **A thin signal.** One win/loss bit per game, spread over ~30 of our moves; which move earned it
   is left for the algorithm to work out (the credit-assignment problem). A search target instead
   gives a graded preference over every legal move at every position. The switch from PPO to Edax
   every-move labels was a switch from the first kind of signal to the second, and the steady
   gains started there.
2. **Optimizing the wrong thing for strength.** PPO improves *sampled* play and, with an entropy
   bonus (`ent_coef=0.03`), keeps the policy spread out. Measured by the top move, our PPO runs made
   the model steadily *worse* (Edax-1 from random openings: 64% after BC, 59% after 5M steps; named
   Edax-1: 78.7% -> 51%), while the sampled numbers looked flat or improving.
3. **Exploiting whatever is cheapest.** Against a near-deterministic Edax from the standard
   opening, the cheapest way to win was to memorize lines (94% vs Edax-6 from the standard opening,
   ~0% from random openings). Self-play has its own versions of this: strategy cycles and forgetting
   earlier lessons.
4. **A value head that could not rank sibling positions.** The PPO value head had a value regret of
   4.7 discs per move (the policy: 1.9), so search with it made play worse. AlphaZero's value head
   learns from millions of outcomes of strong, search-driven games, a much cleaner signal than
   outcomes of weak sampled games.
5. **Scale.** Our PPO runs saw about 30k games per 1M steps, so ~150k games over 5M steps.
   AlphaGo Zero's first run used 4.9M self-play games with 1,600 search simulations per move;
   AlphaZero's chess run used 44M games, generated on thousands of TPUs.

### What it would cost here (rough estimate)

Using numbers measured in this project:

- 1M self-play games x ~60 searched moves per game x 400 simulations per move
  = ~2.4 x 10^10 network evaluations.
- At ~7,400 positions/s (256x12 on the M4 Max, measured at large batch), that is
  ~3.2 x 10^6 s, about **38 days**.
- A 128x8 network (~6x fewer parameters) might bring it to about 1-2 weeks; KataGo-style
  efficiency tricks (cheap, shallow searches on most moves, full searches on a few) cut it further.
- An NVIDIA GPU with bf16 tensor cores would speed exactly this part up (CLAUDE.md, Deferred Work).

Whether 1M games would reach Edax-10 strength is unknown. "Practically achievable" is a question of
compute budget and engineering, not of principle.

### We are closer than it looks

The pipeline built in this project is already most of the AlphaZero loop:

| AlphaZero | This project |
|---|---|
| Self-play generates positions | DAgger collects the positions our model reaches (`collect_positions.py`) |
| Search labels every move | Edax (or Egaroucid) scores every legal move (`label_positions.py --every-move`) |
| Policy trained on search preferences, value on results | Graded policy targets (softmax of move score / 2) and value targets tanh(score / 16) (`edax_train.py`) |
| Search guided by the network | Pruned negamax over the policy's moves, value head at the leaves (`util/search.py`) |

The one part that is not self-taught is **the teacher**. If our own search replaced Edax as the
labeler, the loop would become pure self-improvement. (The exact endgame solver, `solver/endgame.c`,
gives ground truth derived from the rules alone; a purist might call it "search, not knowledge".
AlphaZero didn't use one, but it borrows nothing from other players.)

### A concrete test of the question: a self-teacher round

1. Label a batch of positions with our own combined search instead of Edax.
2. Retrain the network on those labels.
3. Measure whether the network (and the player built on it) gets stronger.

If it does, the model has improved itself with no outside expert, which is the core of the claim.
Caveat: our best network is already partly shaped by Edax, so this tests self-improvement, not
learning from a blank slate. It fits the planned MCTS step: MCTS visit counts are exactly the kind of
target this loop needs.

### Conclusion

The original goal was not wrong. It needed search inside the learning loop, and much more compute
than PPO's early progress suggested. The detour taught us the pieces such a loop is made of.

---

## 2. What is PPO good for?

> It's been a while since I studied the various RL algorithms, but I recall picking PPO because it
> seemed appropriate to the task to me at the time. Give me a ELI5 on what PPO would be good for to
> refresh my memory.

### PPO, explained simply

**Core idea:** try things, see what gets rewarded, do more of what worked, but **change only a
little at a time.**

Picture learning to ride a bike. Each attempt you nudge what you do: lean a bit more, pedal a bit
faster. If a nudge led to staying up longer, you keep it. PPO (Proximal Policy Optimization,
Schulman et al., 2017) adds one rule: *never change your style much in one go*, even if one attempt
went great. A single lucky ride shouldn't make you rebuild your whole technique. That limit is the
**clip range** (0.1 in our setup, `PPO_KWARGS` in `util/util.py`). It is what makes PPO stable, and
stability is why it became the default reinforcement-learning algorithm.

**What it learns:**
- a **policy**, a habit of the form "in this situation, do this, with this probability";
- a **value** estimate (how well things are going), used to judge whether an action turned out
  better or worse than expected (the "advantage").

### What PPO is good for

- **Problems with no teacher and no model of the world**, where you can only act and observe
  rewards: robot locomotion, video games, controlling physical systems.
- **Fast, reflexive decisions**: the result is a policy that runs instantly, with no thinking ahead
  at play time.
- **Continuous or messy action spaces** (steering angles, joint torques), where search-based methods
  struggle.
- **Cheap, plentiful experience**: simulators that run millions of episodes. PPO needs a lot of
  samples but is simple and robust.
- **Fine-tuning a policy that is already decent**, e.g. training language models from human
  feedback (RLHF), where the clipping keeps the model from drifting far from what it already knows.

### Why it seemed right for Reversi, and where it fell short

It was a reasonable choice: it is the standard, well-supported algorithm in Stable Baselines 3, it
supports invalid-action masking (sb3-contrib's MaskablePPO), and it did teach the network to beat
weak opponents (RAI, and its author).

But board games have two features PPO doesn't use:

1. **A perfect model of the world.** The rules are known exactly, so a player can *look ahead*
   (search) instead of learning only from trial and error. PPO ignores this advantage; AlphaZero is
   built around it.
2. **Reward only at the end.** A game yields one win/loss bit, and PPO must work out which of ~30
   moves earned it. Search gives a judgment for every move.

In short: PPO is best for learning reflexes when you can't look ahead. Board games are where you
*can* look ahead, so methods that build search into learning win there.

---

## 3. Have we reinvented AlphaZero?

> So, to a large extent what we've done is reinvent AlphaZero?

### Partly

The closer match is that we rebuilt **AlphaGo** (2016), AlphaZero's predecessor, with a classic
Othello engine's endgame added. AlphaZero is what we get if we close the last gap.

### What we share with the AlphaZero family

- **The network:** a residual trunk with a policy head and a value head (`util/reversi_resnet.py`),
  close to AlphaZero's design.
- **The training targets:** a full preference over every move from a search, not win/loss rewards.
  This is the central idea.
- **Search guided by the network:** the policy chooses which moves to look at; the value head scores
  the leaves.
- **Learning from positions the player itself reaches** (DAgger), which is what self-play data gives
  AlphaZero.

### Where we differ

| | AlphaZero | This project |
|---|---|---|
| Starting knowledge | Rules only | Behavioral cloning on human games (WThor) plus Edax games, as the original AlphaGo started from human games |
| Teacher | Its own search | **Edax** (and a trial of Egaroucid): "expert iteration" with a borrowed expert |
| Search | MCTS | Negamax with policy pruning at every level, closer to classic alpha-beta engines |
| Endgame | The network plays to the end | Exact solver at <= 18 empties, as in Edax, Logistello and Egaroucid |
| Loop | Closed: better network -> better search -> better targets -> ... | Open: the teacher's strength is fixed, so labels can only approach Edax depth 12-16 / Egaroucid level 12 quality |

### What "reinvented" means here

We didn't copy a recipe; each piece came from a measured failure:

| Failure observed | What it led to |
|---|---|
| PPO's top move got steadily worse | Supervised training on search targets (Edax labels) |
| Edax "wins" were memorized lines | Varied starts; evaluation from balanced and named openings |
| The value head couldn't rank sibling positions, so search hurt | Every-move labels; DAgger for the value head |
| Root-only deep search amplified errors in unfamiliar positions | Pruning by the policy at every level |
| Half the lost discs came with <= 20 empties | The exact endgame solver |

These are the same problems AlphaZero's design answers. Arriving at the same answers independently,
through measurement, is good evidence they are the right ones.

### The missing step

Replace Edax with our own search as the teacher, ideally MCTS, and the loop closes: improvement no
longer depends on an outside engine. Steps 21 and 24 also showed a limit of the current open loop:
replacing ~5-8% of the labels with a deeper or stronger teacher moved the network less than we can
measure.

---

## 4. Takeaways and possible next experiments

- **The answer to the original question is yes in principle and demonstrated in practice**, but
  "learning by playing" works when search is inside the learning loop. Outcome-only policy gradients
  (PPO) are the wrong tool for a game whose rules allow look-ahead.
- **The practical obstacles are signal quality and compute**, not theory: about 10^10 network
  evaluations for a serious self-play run, i.e. weeks on this machine and far less on an NVIDIA GPU.
- **The planned order of work already leads there:**
  1. probability-based pruning (step 25, in progress): does smarter search for the same cost help?
  2. MCTS (AlphaZero-style search), if search efficiency is confirmed as the lever;
  3. a **self-teacher round**: label with our own search, retrain, measure. A measurable test of
     whether the model can improve itself without an outside expert;
  4. if that works, a closed self-play loop, possibly on rented NVIDIA hardware, and eventually a
     true blank-slate run (no BC, no Edax) as the purist version of the original goal.
