#!/usr/bin/env python3
"""
Check Loop 1: Tokenizer implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_tokenizer():
    print("🔍 Checking Tokenizer implementation...\n")
    
    try:
        # 尝试导入用户的 tokenizer
        from tokenizer import CharTokenizer
        print("✅ Found CharTokenizer class")
    except ImportError as e:
        print("❌ Cannot import CharTokenizer")
        print("Hint: Make sure you have a tokenizer.py file with CharTokenizer class")
        return False
    
    # 测试 1: 创建 tokenizer
    try:
        tokenizer = CharTokenizer()
        print("✅ Tokenizer created successfully")
        print(f"   Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Failed to create Tokenizer: {e}")
        return False
    
    # 测试 2: encode 功能
    test_text = "hello"
    try:
        tokens = tokenizer.encode(test_text)
        print(f"✅ Encode successful: '{test_text}' -> {tokens}")
        
        # Check if it's a list of numbers
        if not isinstance(tokens, (list, tuple)) or not all(isinstance(t, int) for t in tokens):
            print("⚠️  Warning: encode should return a list of integers")
    except Exception as e:
        print(f"❌ Encode failed: {e}")
        return False
    
    # 测试 3: decode 功能
    try:
        decoded = tokenizer.decode(tokens)
        print(f"✅ Decode successful: {tokens} -> '{decoded}'")
        
        if decoded == test_text:
            print("✅ Perfect! Encode/Decode round trip consistent")
        else:
            print(f"⚠️  Warning: decoded result '{decoded}' doesn't match original '{test_text}'")
    except Exception as e:
        print(f"❌ Decode failed: {e}")
        return False
    
    # 测试 4: 特殊 token
    print("\n🔍 Checking special tokens...")
    special_tokens = ['<pad>', '<unk>', '<eos>']
    found = 0
    
    for token in special_tokens:
        if hasattr(tokenizer, 'char_to_id') and token in tokenizer.char_to_id:
            print(f"✅ Found special token: {token}")
            found += 1
        elif hasattr(tokenizer, 'vocab') and token in tokenizer.vocab:
            print(f"✅ Found special token: {token}")
            found += 1
    
    if found == 0:
        print("⚠️  No special tokens found, but basic functionality is not affected")
    
    # 测试 5: 未知字符处理
    print("\n🔍 Testing unknown character handling...")
    try:
        # 使用一个可能不在 vocab 中的字符
        weird_text = "hello😀world"  # 包含 emoji
        tokens = tokenizer.encode(weird_text)
        decoded = tokenizer.decode(tokens)
        print(f"✅ Handling unknown characters didn't crash")
    except Exception as e:
        print(f"⚠️  Error handling unknown characters, but this is normal: {e}")
    
    print("\n" + "="*50)
    print("🎉 Loop 1 Complete! Tokenizer basic functionality working!")
    print("\nNext: Enter Loop 2 - Transformer")
    return True

if __name__ == "__main__":
    success = test_tokenizer()
    sys.exit(0 if success else 1)