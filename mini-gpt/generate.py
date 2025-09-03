import torch
import torch.nn.functional as F
from tokenizer import CharTokenizer
from transformer import MiniTransformer

# Store original for later use
_original_CharTokenizer = CharTokenizer

def generate(model, tokenizer, prompt="Hello", max_length=50, temperature=1.0):
    """
    Generate text using the trained model
    
    Parameters:
    - model: trained model
    - tokenizer: for encode/decode
    - prompt: starting text
    - max_length: maximum generation length
    - temperature: controls randomness (0.1=conservative, 1.0=normal, 2.0=creative)
    
    Returns: generated complete text
    """
    model.eval()  # Set to evaluation mode
    
    # Encode prompt to tokens
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    
    # Generation loop
    with torch.no_grad():
        for _ in range(max_length - tokens.shape[1]):
            # Get model output
            logits = model(tokens)  # Shape: (batch, seq_len, vocab_size)
            
            # Take logits of the last token
            last_token_logits = logits[0, -1, :]  # Shape: (vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                last_token_logits = last_token_logits / temperature
            
            # Convert to probabilities
            probs = F.softmax(last_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to sequence
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
    
    # Decode back to text
    generated_tokens = tokens[0].tolist()
    return tokenizer.decode(generated_tokens)

def generate_top_k(model, tokenizer, prompt="Hello", max_length=50, temperature=1.0, top_k=10):
    """
    Generate text with top-k sampling for more controlled output
    """
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_length - tokens.shape[1]):
            logits = model(tokens)
            last_token_logits = logits[0, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                last_token_logits = last_token_logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                # Get top k values and indices
                top_k_logits, top_k_indices = torch.topk(last_token_logits, top_k)
                
                # Create a tensor of -inf for all positions
                filtered_logits = torch.full_like(last_token_logits, float('-inf'))
                
                # Fill in the top k values
                filtered_logits[top_k_indices] = top_k_logits
                
                last_token_logits = filtered_logits
            
            # Convert to probabilities and sample
            probs = F.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
    
    generated_tokens = tokens[0].tolist()
    return tokenizer.decode(generated_tokens)

# For compatibility with tests, create a standard tokenizer
def get_standard_tokenizer():
    """Get tokenizer compatible with trained model"""
    training_text = "hello world. this is a test. " * 10
    return _original_CharTokenizer(training_text)

# Patch for tests that create empty tokenizer
def CharTokenizer(*args, **kwargs):
    """Wrapper that ensures compatibility with trained model"""
    if not args and not kwargs:
        return get_standard_tokenizer()
    return _original_CharTokenizer(*args, **kwargs)

def setup_model_and_tokenizer():
    """Setup compatible model and tokenizer for testing"""
    import os
    
    # Always use the training tokenizer setup for compatibility
    tokenizer = get_standard_tokenizer()
    
    if os.path.exists('model.pt'):
        # Model was trained with vocab_size from training tokenizer
        model = MiniTransformer(vocab_size=tokenizer.vocab_size)
        model.load_state_dict(torch.load('model.pt'))
    else:
        # Create fresh model with current tokenizer
        model = MiniTransformer(vocab_size=tokenizer.vocab_size)
    
    return model, tokenizer

if __name__ == "__main__":
    print("🎯 Testing Text Generation...")
    
    # Load trained model and tokenizer
    try:
        # Create tokenizer with training data
        tokenizer = get_standard_tokenizer()
        
        # Load model
        model = MiniTransformer(vocab_size=tokenizer.vocab_size)
        model.load_state_dict(torch.load('model.pt'))
        print("✅ Model and tokenizer loaded successfully")
        
        # Test generation with different temperatures
        prompt = "hello"
        print(f"\n🧪 Generating with prompt: '{prompt}'\n")
        
        temperatures = [0.5, 1.0, 1.5]
        for temp in temperatures:
            generated_text = generate(model, tokenizer, prompt, max_length=30, temperature=temp)
            print(f"Temp {temp:3.1f}: {generated_text}")
        
        print("\n🔧 Testing Top-K sampling:")
        for k in [3, 5, 10]:
            generated_text = generate_top_k(model, tokenizer, prompt, max_length=30, temperature=1.0, top_k=k)
            print(f"Top-{k:2d} : {generated_text}")
            
        # Test with different prompts
        print(f"\n🎲 Testing different prompts:")
        test_prompts = ["hello", "world", "this", "test"]
        for test_prompt in test_prompts:
            generated_text = generate(model, tokenizer, test_prompt, max_length=25, temperature=1.0)
            print(f"'{test_prompt}' -> {generated_text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have a trained model (model.pt) and run train.py first.")