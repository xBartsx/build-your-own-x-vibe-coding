#!/usr/bin/env python3
"""
Final Showcase: Check if all functionality is complete
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def final_showcase():
    print("🎆" * 25)
    print("\n🎉 Mini-GPT Final Showcase 🎉\n")
    print("🎆" * 25)
    print()
    
    results = []
    
    # Check 1: Tokenizer
    print("🔍 Loop 1 - Tokenizer")
    try:
        from tokenizer import CharTokenizer
        tokenizer = CharTokenizer()
        test = tokenizer.decode(tokenizer.encode("hello"))
        if test == "hello":
            print("✅ Tokenizer: Working normally")
            results.append(True)
        else:
            print("❌ Tokenizer: encode/decode inconsistent")
            results.append(False)
    except Exception as e:
        print(f"❌ Tokenizer: {e}")
        results.append(False)
    
    # Check 2: Transformer
    print("\n🔍 Loop 2 - Transformer")
    try:
        import torch
        from transformer import MiniTransformer
        model = MiniTransformer(vocab_size=tokenizer.vocab_size)
        test_input = torch.randint(0, tokenizer.vocab_size, (1, 10))
        with torch.no_grad():
            output = model(test_input)
        if output.shape[2] == tokenizer.vocab_size:
            print("✅ Transformer: Forward pass working")
            results.append(True)
        else:
            print("❌ Transformer: Output dimensions incorrect")
            results.append(False)
    except Exception as e:
        print(f"❌ Transformer: {e}")
        results.append(False)
    
    # Check 3: Training
    print("\n🔍 Loop 3 - Training")
    try:
        import train
        if hasattr(train, 'loss_history') and len(train.loss_history) > 0:
            initial = train.loss_history[0]
            final = train.loss_history[-1]
            if final < initial:
                print(f"✅ Training: Loss decreased from {initial:.2f} to {final:.2f}")
                results.append(True)
            else:
                print("❌ Training: Loss did not decrease")
                results.append(False)
        else:
            print("⚠️  Training: No loss history found")
            results.append(False)
    except:
        print("⚠️  Training: train.py not found")
        results.append(False)
    
    # Check 4: Generation
    print("\n🔍 Loop 4 - Generation")
    try:
        # Try importing from different places
        try:
            from generate import generate
        except:
            from train import generate
        
        # Load trained model (if available)
        if os.path.exists('model.pt'):
            model.load_state_dict(torch.load('model.pt'))
        
        text = generate(model, tokenizer, "Hello", max_length=30)
        if len(text) > 5:
            print(f"✅ Generation: Can generate text")
            print(f"   Example: '{text[:50]}'")
            results.append(True)
        else:
            print("❌ Generation: Generation failed")
            results.append(False)
    except Exception as e:
        print(f"❌ Generation: {e}")
        results.append(False)
    
    # Summary
    print("\n" + "="*50)
    print("\n🏆 Final Results 🏆\n")
    
    completed = sum(results)
    total = len(results)
    
    checkmarks = [
        "✅ Tokenization successful" if results[0] else "❌ Tokenization failed",
        "✅ Forward pass working" if results[1] else "❌ Forward pass failed",
        "✅ Loss decreasing" if results[2] else "❌ Loss not decreasing",
        "✅ Can generate text" if results[3] else "❌ Cannot generate text"
    ]
    
    for check in checkmarks:
        print(f"  {check}")
    
    if completed == total:
        print("\n🎆" * 25)
        print("\n🎉 Congratulations! You have successfully built your own Mini-GPT!")
        print("✅ You now understand the core principles of GPT!")
        print("\n🎆" * 25)
    else:
        print(f"\n📊 Completion: {completed}/{total} ({completed*100//total}%)")
        print("\n💪 Keep going! You're almost there!")
        print("💡 Hint: Run the test scripts for each Loop to see specific issues")
    
    # Next step suggestions
    print("\n🚀 Next you could try:")
    print("  1. Train with larger texts (Shakespeare, novels, etc.)")
    print("  2. Implement BPE tokenizer")
    print("  3. Add attention visualization")
    print("  4. Implement beam search")
    print("  5. Create a web interface")
    
    return completed == total

if __name__ == "__main__":
    success = final_showcase()
    sys.exit(0 if success else 1)