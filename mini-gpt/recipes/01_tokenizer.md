# Prompt Recipe: Building Character-Level Tokenizer

## Copy the content below directly to LLM:

---

I want to build a simple character-level tokenizer with the following requirements:

1. Create a CharTokenizer class
2. Implement `encode(text)` method: text -> list of numbers
3. Implement `decode(tokens)` method: list of numbers -> text
4. Include special tokens: `<pad>`, `<unk>`, `<eos>`
5. Use dictionary to store char -> id mapping

Example behavior:
- encode("hello") -> [7, 4, 11, 11, 14]
- decode([7, 4, 11, 11, 14]) -> "hello"

Please give me complete runnable Python code including:
- Class definition
- A simple test example
- Print vocab size

Code style: Simple and direct, with comments explaining key steps.

---

## Usage Tips:
1. If LLM's code is too complex, ask to "simplify to under 50 lines"
2. If there are runtime errors, paste the error messages back
3. After success, run `python checks/01_tokenizer_test.py` to verify