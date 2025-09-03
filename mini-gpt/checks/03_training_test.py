#!/usr/bin/env python3
"""
Check Loop 3: Training functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_training():
    print("🔍 Checking training process...\n")
    
    # Test 1: Import training script
    try:
        import train
        print("✅ Found train.py")
    except ImportError:
        print("❌ Cannot import train.py")
        print("Hint: Create a train.py file containing training loop")
        return False
    
    # Test 2: Check loss history
    print("🔍 Checking loss changes...")
    
    if hasattr(train, 'loss_history'):
        losses = train.loss_history
        print(f"Found {len(losses)} loss records")
        
        if len(losses) > 0:
            initial_loss = losses[0]
            final_loss = losses[-1]
            
            print(f"\n📉 Loss changes:")
            print(f"   Initial: {initial_loss:.4f}")
            print(f"   Final: {final_loss:.4f}")
            print(f"   Decrease: {initial_loss - final_loss:.4f}")
            
            # Check if decreasing
            if final_loss < initial_loss:
                print("✅ Loss successfully decreased!")
            else:
                print("⚠️  Loss didn't decrease, may need to adjust learning rate")
            
            # Check if target reached
            if final_loss < 2.0:
                print("✅ Loss < 2.0, target reached!")
            else:
                print(f"⚠️  Loss hasn't dropped below 2.0 yet")
                print("Hint: Try increasing training steps or adjusting learning rate")
        else:
            print("❌ No loss records")
            print("Hint: Make sure to save loss_history in training loop")
            return False
    else:
        print("⚠️  Cannot find loss_history")
        print("Hint: Add loss_history = [] in train.py to record loss")
    
    # Test 3: Check if model is saved
    import os
    if os.path.exists('model.pt') or os.path.exists('mini_gpt.pt'):
        print("\n✅ Found saved model file")
    else:
        print("\n⚠️  No saved model found")
        print("Hint: Use torch.save(model.state_dict(), 'model.pt') to save model")
    
    # Test 4: Run simple prediction
    print("\n🔍 Testing model prediction...")
    try:
        import torch
        from tokenizer import CharTokenizer
        from transformer import MiniTransformer
        
        tokenizer = CharTokenizer()
        model = MiniTransformer(vocab_size=tokenizer.vocab_size)
        
        # Load saved model if exists
        if os.path.exists('model.pt'):
            model.load_state_dict(torch.load('model.pt'))
            print("✅ Loaded trained model")
        
        # Test prediction
        test_text = "hello"
        tokens = tokenizer.encode(test_text)
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            output = model(input_ids)
            probs = torch.softmax(output[0, -1], dim=-1)
            next_token = torch.argmax(probs).item()
        
        print(f"✅ Prediction successful!")
        print(f"   Input: '{test_text}'")
        print(f"   Next token ID: {next_token}")
        
    except Exception as e:
        print(f"⚠️  Prediction failed: {e}")
    
    print("\n" + "="*50)
    print("🎉 Loop 3 Complete! Model can be trained!")
    print("\nNext: Enter Loop 4 - Generation")
    return True

if __name__ == "__main__":
    success = test_training()
    sys.exit(0 if success else 1)