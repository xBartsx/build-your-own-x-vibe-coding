# 🤖 Mini-GPT: Build Your First Language Model from Scratch

> Build a "talking" AI with 5 micro-loops (2.5 hours)

## 🎯 What You'll Learn
- **Tokenization**: How to convert text into numbers
- **Attention**: The core mechanism of Transformers
- **Training**: How to make models learn to predict the next word
- **Generation**: Making models generate text

## 🚀 Project Structure
```
mini-gpt/
├── recipes/          # Prompt templates (copy-paste ready)
├── checks/           # Automated test scripts (verify each step)
├── scaffolds/        # Skeleton code (directly runnable)
├── docs/            # Additional documentation
└── vibes.md         # Your build diary
```

## 📚 Learning Path (5 Micro-Loops)

### Loop 1: Tokenizer (25 minutes)
**Goal**: Convert "hello world" to [123, 456]

**What you'll do**:
1. Copy the prompt from `recipes/01_tokenizer.md`
2. Get an LLM to help you write a simple character-level tokenizer
3. Run `checks/01_tokenizer_test.py` to verify
4. Record in `vibes.md`: What worked? What got stuck?

**Checkpoint**: 
```bash
python checks/01_tokenizer_test.py
# ✅ Should see: "hello" -> [7, 4, 11, 11, 14]
```

---

### Loop 2: Mini Transformer (25 minutes)
**Goal**: Implement a 2-layer attention mechanism

**What you'll do**:
1. Start from `scaffolds/02_transformer_skeleton.py`
2. Use `recipes/02_transformer.md` prompt to fill in the code
3. Run `checks/02_transformer_test.py`
4. Record in `vibes.md`

**Checkpoint**:
```bash
python checks/02_transformer_test.py
# ✅ Should see: "Input shape: (1, 10, 64), Output shape: (1, 10, 64)"
```

---

### Loop 3: Training Loop (25 minutes)
**Goal**: Make the model learn to predict "hello worl[d]"

**What you'll do**:
1. Use `recipes/03_training.md` to set up the training loop
2. Train on small text for 100 steps
3. Run `checks/03_training_test.py`
4. Record the loss decrease

**Checkpoint**:
```bash
python checks/03_training_test.py
# ✅ Should see: "Loss decreased from 3.5 to < 2.0"
```

---

### Loop 4: Text Generation (20 minutes)
**Goal**: Make the model continue sentences

**What you'll do**:
1. Use `recipes/04_generation.md` to implement sampling
2. Given "Once upon", make the model generate 10 words
3. Run `checks/04_generation_test.py`

**Checkpoint**:
```bash
python checks/04_generation_test.py
# ✅ Should see the model outputting text (even if it's gibberish)
```

---

### Loop 5: Polish & Play (25 minutes)
**Goal**: Fine-tune and play with your model

**What you'll do**:
1. Train with larger data (Shakespeare text)
2. Adjust temperature parameters to see effects
3. Run final check `checks/05_final_test.py`
4. Screenshot and share your "AI poet"

---

## 🛠️ Environment Setup

```bash
# Minimal installation (only need these)
pip install numpy torch

# Optional: if you want visualization
pip install matplotlib
```

## 💡 What if I Get Stuck?

1. **Every Loop has an escape route**:
   - Check reference implementations in `scaffolds/`
   - Run working versions first, understand then write your own

2. **Prompt not working?**
   - `recipes/` has backup prompts
   - Each prompt has been tested at least 5 times

3. **Check failed?**
   - Every `checks/*.py` has detailed error messages
   - Tells you exactly which line needs fixing

## 🎉 Completion Marker

Run `python checks/final_showcase.py`, you should see:
```
🎊 Congratulations! Your Mini-GPT can:
✓ Tokenize successfully
✓ Forward propagation working
✓ Loss is decreasing
✓ Generate text
✓ You now understand the core principles of GPT!
```

## 📝 Example vibes.md

```markdown
## Loop 1 - Tokenizer (10:30-10:55)
- Intent: Understand how text becomes numbers
- Friction: Initially didn't understand why <unk> token is needed
- Next: Try BPE tokenizer?

## Loop 2 - Transformer (11:00-11:25)  
- Intent: Implement attention mechanism
- Friction: Matrix dimensions kept mismatching, debugged for 15 minutes
- Next: Visualize attention weights
```

## 🚦 Let's Begin!

```bash
cd mini-gpt
cp recipes/01_tokenizer.md my_prompt.md
# Open your favorite LLM and start Loop 1!
```

---

*Remember the core of Vibe-Coding: Don't seek perfection, but ensure each loop runs. Get it working first, then optimize!*