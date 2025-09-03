"""
Mini Transformer Scaffold
This is skeleton code, you need to fill in the TODO parts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=100, d_model=64, n_heads=4, n_layers=2, max_len=512):
        super().__init__()
        self.d_model = d_model
        
        # TODO: Implement these layers
        # self.token_embedding = nn.Embedding(???, ???)
        # self.position_embedding = nn.Embedding(???, ???)
        # self.layers = nn.ModuleList([??? for _ in range(n_layers)])
        # self.ln_final = nn.LayerNorm(???)
        # self.lm_head = nn.Linear(???, ???)
        
        # Hints:
        # - token_embedding: vocab_size -> d_model
        # - position_embedding: max_len -> d_model
        # - each layer contains attention and feed forward
        # - lm_head: d_model -> vocab_size
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # TODO: Implement forward pass
        # 1. Get token embeddings
        # 2. Add position embeddings
        # 3. Pass through all transformer layers
        # 4. Apply final layer norm
        # 5. Pass through lm_head to get logits
        
        # x = self.token_embedding(input_ids)
        # positions = torch.arange(seq_len, device=input_ids.device)
        # x = x + self.position_embedding(positions)
        # ...
        
        pass  # Remove this line and implement the logic above


class TransformerBlock(nn.Module):
    """A Transformer block: attention + feed forward network"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        # TODO: Implement
        # self.attention = nn.MultiheadAttention(???, ???)
        # self.ln1 = nn.LayerNorm(???)
        # self.ffn = FeedForward(???)
        # self.ln2 = nn.LayerNorm(???)
        pass
        
    def forward(self, x):
        # TODO: Implement residual connections
        # x = x + self.attention(...)
        # x = self.ln1(x)
        # x = x + self.ffn(x)
        # x = self.ln2(x)
        pass


class FeedForward(nn.Module):
    """Feed forward network: two-layer MLP"""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model  # Usually 4x
        # TODO: Implement two-layer MLP
        # self.fc1 = nn.Linear(???, ???)
        # self.fc2 = nn.Linear(???, ???)
        pass
        
    def forward(self, x):
        # TODO: fc1 -> relu -> fc2
        pass


# Test code (do not modify)
if __name__ == "__main__":
    # Create model
    model = MiniTransformer(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    
    # Create input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Forward pass
    try:
        output = model(input_ids)
        print(f"✅ Forward pass successful!")
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected: ({batch_size}, {seq_len}, 100)")
        
        if output.shape == (batch_size, seq_len, 100):
            print("✅ Shape correct!")
        else:
            print("❌ Shape incorrect, check your implementation")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Hint: Check if all TODOs are implemented")