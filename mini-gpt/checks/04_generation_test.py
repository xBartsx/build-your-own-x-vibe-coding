#!/usr/bin/env python3
"""
Check Loop 4: Generation text generation functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_generation():
    print("🔍 Checking text generation functionality...\n")
    
    # Test 1: Import generate function
    try:
        from generate import generate
        print("✅ Found generate function")
    except ImportError:
        try:
            from train import generate
            print("✅ Found generate function in train.py")
        except ImportError:
            print("❌ Cannot find generate function")
            print("Hint: Create generate.py or add generate function to train.py")
            return False
    
    # Prepare model and tokenizer
    try:
        import torch
        from tokenizer import CharTokenizer
        from transformer import MiniTransformer
        
        tokenizer = CharTokenizer()
        model = MiniTransformer(vocab_size=tokenizer.vocab_size)
        
        # Try to load trained model
        if os.path.exists('model.pt'):
            model.load_state_dict(torch.load('model.pt'))
            print("✅ Loaded trained model")
        else:
            print("⚠️  Using untrained model (output may be random)")
        
        model.eval()
        
    except Exception as e:
        print(f"❌ Model preparation failed: {e}")
        return False
    
    # Test 2: Basic generation
    print("\n🔍 Testing basic generation...")
    try:
        prompt = "hello"
        generated = generate(model, tokenizer, prompt=prompt, max_length=20)
        
        print(f"✅ Generation successful!")
        print(f"   Prompt: '{prompt}'")
        print(f"   Generated: '{generated}'")
        
        # Check if new content was actually generated
        if len(generated) > len(prompt):
            print(f"✅ Generated {len(generated) - len(prompt)} new characters")
        else:
            print("⚠️  No new content generated")
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        print("Hint: Check generate function parameters")
        return False
    
    # Test 3: Different temperatures
    print("\n🔍 Testing different temperatures...")
    temperatures = [0.5, 1.0, 1.5]
    outputs = []
    
    for temp in temperatures:
        try:
            generated = generate(
                model, tokenizer, 
                prompt="Once", 
                max_length=30,
                temperature=temp
            )
            outputs.append(generated)
            print(f"✅ Temperature {temp}: '{generated[:50]}...'")
        except Exception as e:
            print(f"⚠️  Temperature {temp} failed: {e}")
    
    # Check diversity
    if len(set(outputs)) > 1:
        print("✅ Different temperatures produced different results (good!)")
    else:
        print("⚠️  All temperature results are the same (temperature may not be implemented)")
    
    # Test 4: Long text generation
    print("\n🔍 Testing long text generation...")
    try:
        long_text = generate(
            model, tokenizer,
            prompt="The",
            max_length=100
        )
        print(f"✅ Generated {len(long_text)} characters")
        
        # Show first 100 characters
        print(f"\n📝 Generation example:")
        print(f"'{long_text[:100]}...'")
        
    except Exception as e:
        print(f"⚠️  Long text generation failed: {e}")
    
    print("\n" + "="*50)
    print("🎉 Loop 4 complete! Your Mini-GPT can generate text!")
    print("\nNext step: Enter Loop 5 - Polish & Play")
    return True

if __name__ == "__main__":
    success = test_generation()
    sys.exit(0 if success else 1)