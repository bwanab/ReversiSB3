# Session Notes - 2025-12-10

## Web Interface - COMPLETED ✅

Successfully built web interface for playing against the model:
- Flask backend at `web_play.py`
- Visual board with click-to-move
- Model can play as Black or White
- Run with: `python web_play.py -m bc_pretrained`

### Fixes Applied:
1. **Fixed env.step() dependency**: Switched to low-level methods (`get_next_state`, `has_valid`, `get_winner`)
2. **Fixed player switching bug**: `get_next_state()` already switches player internally - removed redundant switches
3. **UX improvements**:
   - Brighter last-move highlight (cyan `#00bcd4`)
   - Fixed column label alignment (A-H now align with columns)

## Model Performance Against Piccolo iPhone App

**Result**: Model couldn't win against Piccolo AI levels 3-4 (out of 10)

### Observations:
- Model made questionable moves leading to corner captures by opponent
- In one game: captured all 4 corners but still lost
- Suggests lack of understanding of:
  - Parity (controlling last moves)
  - Mobility vs. disk count balance
  - Tempo (when to take corners)
  - Edge control beyond corners

### Analysis: Overfitting to RAI's Style

**Current training distribution**:
- 6M timesteps mixed mode: 75% self-play, 20% random, 5% RAI
- Alternating RAI depth 1 and 2
- BC pre-training from RAI depth-2 games

**The problem**: Model learned "anti-RAI tactics" that don't generalize
- RAI uses minimax with specific heuristics
- Piccolo likely uses different engine/evaluation
- Model exploits RAI's weaknesses, doesn't learn Reversi fundamentals
- 86% vs RAI-2 is impressive but specialized, not general

**The "4 corners but lost" tells us**:
- Model over-prioritizes corners (maybe RAI does too)
- Doesn't understand deeper concepts like parity, mobility sacrifice, endgame tempo
- RAI doesn't punish these mistakes, but other opponents do

## Next Steps - Discussion

### What WON'T Help:
- **More pure self-play**: Would reinforce current anti-RAI strategy, no exposure to different styles

### What WOULD Help:

#### 1. Diverse Opponent Training (Highest Priority)
Train against multiple opponent types:
- RAI at multiple depths (1, 2, 3, 4) with varied ratios
- Example: 40% RAI-2, 20% RAI-3, 20% self-play, 20% RAI-1
- Forces model to learn flexible strategies

#### 2. BC from Diverse Sources
Generate new BC datasets from:
- Multiple engines (if available)
- Different RAI configurations
- WThor database (public expert Othello games)
- Piccolo games (if extractable)

#### 3. Curriculum Learning
- Start RAI depth 1 → 2 → 3 → 4
- Mix in other opponents as model improves
- Don't let it specialize too early

#### 4. Train Against Stronger Opponents
- **Edax**: Much stronger than RAI, would teach better fundamentals
- Even Edax level 1-5 would provide different strategic patterns

## Edax Python Wrapper Idea

**Goal**: Create reusable Python library for interfacing with Edax
- Primary benefit: Train against Edax at various levels
- Secondary benefit: Useful for broader Python/Reversi community

**Feasibility**: ✅ HIGHLY FEASIBLE
- Don't need to modify Edax C code
- Edax has text-based command protocol
- Use subprocess to communicate (like python-chess does with Stockfish)
- Standard architecture: Python → stdin → Edax process → stdout → Python

**Current Blocker**: 🚧
- Edax binary compiled successfully at `~/src/edax-reversi/bin/edax`
- But Edax requires `data/eval.dat` to run
- Downloaded `eval.7z` from GitHub releases
- Can't extract: needs `p7zip` (brew install blocked for user)
- **Need alternative extraction method**

### Wrapper Design (Planned):

```python
class EdaxEngine:
    def __init__(self, level=5, executable_path='edax')
    def set_level(self, level)
    def set_board(self, board)
    def get_move(self, board=None)
    def evaluate(self, board)
    def quit()

# Integration with existing opponents
def edax_action(env, state, level=5):
    # Compatible with util/opponents.py interface
```

## Action Items

### Immediate (User):
- [ ] Extract eval.7z to get eval.dat (find p7zip alternative or extract on another machine)
- [ ] Place eval.dat in `~/src/edax-reversi/data/`
- [ ] Test Edax runs: `cd ~/src/edax-reversi/bin && ./edax`

### Next Session (With Claude):
- [ ] Test Edax interactive commands to understand protocol
- [ ] Build basic EdaxEngine Python wrapper
- [ ] Integrate as opponent in training system
- [ ] Test training against Edax level 1-5
- [ ] Evaluate if model improves against Piccolo

## Key Insight

This is a **success story with an important lesson**:
- ✅ BC + RL approach worked brilliantly for the specified goal (beat RAI)
- ✅ Model learned sophisticated strategy (86% vs RAI-2)
- ⚠️ But discovered the model specialized rather than generalized
- 📚 Lesson: Training distribution matters as much as training algorithm

Like training a chess player exclusively against one opponent - amazing at beating that person, struggles against different styles.

## Files Modified This Session

- `web_play.py` - Created Flask web interface
- `templates/index.html` - HTML template for UI
- `static/style.css` - Styling (last-move highlight, label alignment)
- `static/script.js` - Frontend game logic
- This file: `session_notes.md`

## Performance Summary

**Model: bc_pretrained (6M timesteps BC + RL)**
- vs RAI-1: 87%
- vs RAI-2: 86%
- vs RAI-3: 20%
- vs Random: 98%
- vs Piccolo AI-3/4: 0% (couldn't win a single game)

**Diagnosis**: Model is "RAI-specialized" not "Reversi-general"
