# Prompt Recipe: Implementing Mini Transformer

## Copy directly to LLM:

---

I have a Transformer skeleton code that needs core implementation:

```python
import torch
import torch.nn as nn

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=100, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        # TODO: implement embedding, attention, output layers
        
    def forward(self, x):
        # TODO: implement forward propagation
        pass
```

Requirements:
1. **Embedding layer**: token embedding + positional embedding
2. **Self-Attention**: simplified multi-head attention (no mask)
3. **Feed Forward**: two Linear layers + ReLU
4. **Layer Norm**: after each sub-layer
5. Output shape: (batch, seq_len, vocab_size)

Simplification requirements:
- No dropout
- No complex positional encoding, use learnable ones
- Keep code under 80 lines

Please give me complete implementation and a simple test:
```python
# Test
model = MiniTransformer()
input_ids = torch.randint(0, 100, (1, 10))  # batch=1, seq_len=10
output = model(input_ids)
print(f"Output shape: {output.shape}")  # Should be (1, 10, 100)
```

---

## Backup Prompt (if first one doesn't work):

"Give me the simplest GPT-style Transformer, just needs forward propagation, under 50 lines of code"