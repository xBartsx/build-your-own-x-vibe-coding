class CharTokenizer:
    def __init__(self, text_corpus=None):
        # Special tokens
        self.special_tokens = ['<pad>', '<unk>', '<eos>']
        
        # Initialize mappings with special tokens
        self.char_to_id = {}
        self.id_to_char = {}
        
        # Always add special tokens
        for i, token in enumerate(self.special_tokens):
            self.char_to_id[token] = i
            self.id_to_char[i] = token
        
        # Build vocabulary from corpus if provided, or use default
        if text_corpus is None:
            # Default training text for compatibility
            text_corpus = "hello world. this is a test. " * 10
        self._build_vocab(text_corpus)
    
    def _build_vocab(self, text_corpus):
        # Add unique characters from corpus (special tokens already added in __init__)
        unique_chars = sorted(set(text_corpus))
        for char in unique_chars:
            if char not in self.char_to_id:
                char_id = len(self.char_to_id)
                self.char_to_id[char] = char_id
                self.id_to_char[char_id] = char
    
    def encode(self, text):
        # Convert text to list of token IDs
        tokens = []
        for char in text:
            if char in self.char_to_id:
                tokens.append(self.char_to_id[char])
            else:
                # Use <unk> token for unknown characters
                tokens.append(self.char_to_id['<unk>'])
        return tokens
    
    def decode(self, tokens):
        # Convert list of token IDs back to text
        chars = []
        for token_id in tokens:
            if token_id in self.id_to_char:
                chars.append(self.id_to_char[token_id])
            else:
                chars.append('<unk>')
        return ''.join(chars)
    
    @property
    def vocab_size(self):
        """Return vocabulary size"""
        return len(self.char_to_id)


# Test example
if __name__ == "__main__":
    # Create tokenizer with a sample corpus
    corpus = "hello world this is a test"
    tokenizer = CharTokenizer(corpus)
    
    # Test encoding and decoding
    text = "hello"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original text: '{text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Show vocabulary mapping
    print("\nVocabulary mapping:")
    for char, char_id in sorted(tokenizer.char_to_id.items(), key=lambda x: x[1]):
        print(f"'{char}' -> {char_id}")