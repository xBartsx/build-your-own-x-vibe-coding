#!/usr/bin/env python3
"""
Check Loop 2: Transformer implementation
"""

import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_transformer():
    print("🔍 Checking Mini Transformer implementation...\n")
    
    # Test 1: Import
    try:
        from transformer import MiniTransformer
        print("✅ Found MiniTransformer class")
    except ImportError:
        print("❌ Cannot import MiniTransformer")
        print("Hint: Make sure you have transformer.py file")
        return False
    
    # Test 2: Create model
    try:
        model = MiniTransformer(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2
        )
        print("✅ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test 3: Forward propagation
    print("\n🔍 Testing forward propagation...")
    batch_size = 2
    seq_len = 10
    
    try:
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        print(f"Input shape: {input_ids.shape}")
        
        with torch.no_grad():
            output = model(input_ids)
        
        print(f"✅ Forward propagation successful")
        print(f"Output shape: {output.shape}")
        
        # Check shape
        expected_shape = (batch_size, seq_len, 100)
        if output.shape == expected_shape:
            print(f"✅ Output shape correct: {output.shape}")
        else:
            print(f"❌ Shape wrong. Expected: {expected_shape}, Actual: {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Forward propagation failed: {e}")
        print("Hint: Check your forward method implementation")
        return False
    
    # Test 4: Check if output is reasonable
    print("\n🔍 Checking output reasonableness...")
    
    # Check for NaN
    if torch.isnan(output).any():
        print("❌ Output contains NaN!")
        return False
    else:
        print("✅ Output has no NaN")
    
    # Check for Inf
    if torch.isinf(output).any():
        print("❌ Output contains Inf!")
        return False
    else:
        print("✅ Output has no Inf")
    
    # Test 5: Different input lengths
    print("\n🔍 Testing different sequence lengths...")
    for test_len in [1, 5, 20]:
        try:
            test_input = torch.randint(0, 100, (1, test_len))
            with torch.no_grad():
                test_output = model(test_input)
            print(f"✅ Sequence length {test_len}: OK")
        except Exception as e:
            print(f"❌ Sequence length {test_len} failed: {e}")
    
    print("\n" + "="*50)
    print("🎉 Loop 2 Complete! Transformer basic structure correct!")
    print("\nNext: Enter Loop 3 - Training")
    return True

if __name__ == "__main__":
    success = test_transformer()
    sys.exit(0 if success else 1)