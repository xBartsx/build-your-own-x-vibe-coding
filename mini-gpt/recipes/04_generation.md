# Prompt Recipe: Text Generation

## Copy directly to LLM:

---

I already have a trained Mini Transformer, now want to make it generate text.

Need to implement a `generate()` function:

```python
def generate(model, tokenizer, prompt="Hello", max_length=50, temperature=1.0):
    """
    Parameters:
    - model: trained model
    - tokenizer: for encode/decode
    - prompt: starting text
    - max_length: maximum generation length
    - temperature: controls randomness (0.1=conservative, 1.0=normal, 2.0=creative)
    
    Returns: generated complete text
    """
    pass
```

Implementation requirements:

1. **Sampling method**:
   - Use temperature sampling: logits / temperature
   - Use torch.multinomial() to sample from probability distribution
   - Optional: implement top-k sampling (only choose from top k most likely tokens)

2. **Generation flow**:
   ```python
   # 1. encode prompt
   tokens = tokenizer.encode(prompt)
   
   # 2. generation loop
   for _ in range(max_length - len(tokens)):
       # get model output
       # take logits of last token
       # apply temperature
       # sample next token
       # add to sequence
       
   # 3. decode back to text
   return tokenizer.decode(tokens)
   ```

3. **Test example**:
   ```python
   # Effects of different temperatures
   for temp in [0.5, 1.0, 1.5]:
       text = generate(model, tokenizer, "Once upon", temperature=temp)
       print(f"Temp {temp}: {text}")
   ```

Please give me complete generate function and test code.

---

## Backup Tips:
- If generating repetition: try increasing temperature
- If generating gibberish: check if tokenizer decode is correct
- If too slow: wrap with `with torch.no_grad()`