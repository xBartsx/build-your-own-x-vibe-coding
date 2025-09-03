# Prompt Recipe: Training Loop

## Copy directly to LLM:

---

I already have tokenizer and transformer model, now need to train it.

Please help me write a training script:

1. **Data preparation**:
   - Use simple text (like "hello world. this is a test. " repeated 10 times)
   - Create training data: input is first n-1 tokens, target is last n-1 tokens

2. **Training loop**:
   ```python
   for epoch in range(100):
       # forward propagation
       # calculate cross entropy loss
       # backward propagation
       # print loss (every 10 steps)
   ```

3. **Checkpoints**:
   - Initial loss should be around 4.6 (log(100))
   - After 100 steps loss should decrease to below 2.0
   - Save loss history for plotting

4. **Complete code structure**:
   ```python
   import torch
   import torch.nn as nn
   from tokenizer import CharTokenizer  # assume exists
   from transformer import MiniTransformer  # assume exists
   
   # Initialize
   tokenizer = CharTokenizer()
   model = MiniTransformer(vocab_size=tokenizer.vocab_size)
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
   # Prepare data
   text = "hello world. " * 10
   # ... complete the rest
   ```

Please give me complete runnable code.

---

## Debugging Tips:
- If loss doesn't decrease: check learning rate, try 0.01
- If loss is NaN: learning rate too high, change to 0.0001
- If dimension error: check input/target shapes