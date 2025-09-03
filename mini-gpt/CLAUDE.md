# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Mini-GPT implementation tutorial that follows the Vibe-Coding approach - building a simple language model from scratch in 5 loops (2.5 hours). The project teaches core concepts like tokenization, attention mechanisms, training loops, and text generation through hands-on implementation.

## Development Commands

### Environment Setup
```bash
# Required dependencies
pip install numpy torch

# Optional visualization
pip install matplotlib
```

### Testing Each Loop
```bash
# Loop 1: Test tokenizer implementation
python checks/01_tokenizer_test.py

# Loop 2: Test transformer architecture
python checks/02_transformer_test.py

# Loop 3: Test training functionality
python checks/03_training_test.py

# Loop 4: Test text generation
python checks/04_generation_test.py

# Final validation of all components
python checks/final_showcase.py
```

## Code Architecture

### Implementation Flow
The project follows a progressive implementation path where each loop builds on the previous:

1. **Tokenizer** (`tokenizer.py`): Character-level tokenization with special tokens (`<pad>`, `<unk>`, `<eos>`)
2. **Transformer** (`transformer.py`): Mini transformer with attention layers, position embeddings, and feed-forward networks
3. **Training** (`train.py`): Training loop with loss calculation and optimization
4. **Generation** (`generate.py` or in `train.py`): Text generation with temperature-based sampling

### Key Components

**CharTokenizer**: Converts text to token IDs and back
- `encode(text)`: text → token IDs
- `decode(tokens)`: token IDs → text
- Maintains vocabulary with special tokens

**MiniTransformer**: 2-layer transformer model
- Token and position embeddings
- Multi-head attention blocks
- Feed-forward networks with residual connections
- Language modeling head for next-token prediction

**Training Process**: 
- Uses cross-entropy loss for next-token prediction
- Tracks loss history for monitoring
- Saves trained model to `model.pt`

## Development Guidelines

### Using Recipe Prompts
Each loop has a corresponding recipe in `recipes/` that provides exact prompts to give to LLMs for implementation help. These are tested prompts that work reliably.

### Using Scaffolds
The `scaffolds/` directory contains skeleton code with TODOs marking where implementation is needed. Use `scaffolds/02_transformer_skeleton.py` as a starting point for the transformer implementation.

### Progress Tracking
Update `vibes.md` after each loop with:
- **意图** (Intent): What you aimed to accomplish
- **摩擦** (Friction): Challenges encountered
- **下一步** (Next Step): What comes next

### Common Issues and Solutions
- Matrix dimension mismatches: Usually in position embeddings or attention layers
- Loss not decreasing: Adjust learning rate (typically 0.001 to 0.01)
- Generation issues: Ensure `model.eval()` and `torch.no_grad()` are used
- OOM errors: Use `torch.no_grad()` during inference

### Expected Outputs
- Loop 1: `"hello"` → `[7, 4, 11, 11, 14]` (exact values may vary)
- Loop 2: Input shape `(1, 10, 64)` → Output shape `(1, 10, vocab_size)`
- Loop 3: Loss should decrease from ~3.5 to <2.0 within 100 steps
- Loop 4: Model should generate text (even if nonsensical initially)