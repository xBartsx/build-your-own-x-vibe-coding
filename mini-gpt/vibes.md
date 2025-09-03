# 📝 Mini-GPT Build Diary

> Record 3 key points for each loop: Intent → Friction → Next Step

---

## Loop 1: Tokenizer (10:00-10:25)
- **Intent**: Understand how text becomes numbers
- **Friction**: Initially didn't understand why `<unk>` token was needed, later realized handling unknown characters is important
- **Next Step**: Implement Transformer structure

## Loop 2: Transformer (10:30-10:55)
- **Intent**: Build attention mechanism
- **Friction**: Matrix dimensions kept mismatching, debugged for 15 minutes, finally found it was a position embedding issue
- **Next Step**: Make it learn to predict the next word

## Loop 3: Training (11:00-11:25) 
- **Intent**: Make the model learn to predict
- **Friction**: Loss wouldn't decrease! Adjusted learning rate for ages, had to change from 0.001 to 0.01 before it started dropping
- **Next Step**: Implement text generation

## Loop 4: Generation (11:30-11:50)
- **Intent**: Make the model "talk"
- **Friction**: Temperature concept was unclear, looked at several examples before understanding it controls randomness
- **Next Step**: Train with larger dataset

## Loop 5: Polish (11:55-12:20)
- **Intent**: Train with Shakespeare text
- **Friction**: Training too slow! 100MB text takes forever to run
- **Next Step**: Implement BPE tokenizer for better efficiency

---

## 🎆 Summary

**Biggest Takeaways**:
- Transformer is actually not complex, core is just "attention"
- Writing from scratch feels better than reading papers
- Debugging is the best way to learn

**Unexpected Discoveries**:
- Small data can still train interesting patterns
- temperature = 2.0 produces hilarious results

**Pitfalls Encountered**:
1. Forgetting model.eval() caused unstable generation results
2. Got batch dimensions confused
3. Missing torch.no_grad() caused OOM

**Next Project Ideas**:
- Mini BERT (bidirectional encoding)
- Mini Diffusion Model
- Implement RLHF from scratch