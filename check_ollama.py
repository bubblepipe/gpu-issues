#!/usr/bin/env python3
"""Check Ollama configuration on remote server."""

import subprocess
import json
import sys

def check_ollama():
    """Check if Ollama is running and what models are available."""
    
    print("Checking Ollama on h100 server...")
    print("-" * 50)
    
    # Check if Ollama is running
    print("\n1. Checking if Ollama service is running...")
    result = subprocess.run(
        "ssh h100 'curl -s http://localhost:11434/api/tags'",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Failed to connect to Ollama: {result.stderr}")
        print("\nTry running on h100: ollama serve")
        return False
    
    try:
        data = json.loads(result.stdout)
        models = data.get('models', [])
        
        print(f"✅ Ollama is running!")
        print(f"\n2. Available models ({len(models)}):")
        
        for model in models:
            name = model.get('name', 'unknown')
            size = model.get('size', 0) / (1024**3)  # Convert to GB
            print(f"   - {name} ({size:.1f} GB)")
        
        # Check specifically for qwen3:235b
        qwen_models = [m for m in models if 'qwen' in m.get('name', '').lower()]
        if qwen_models:
            print(f"\n✅ Found Qwen models: {[m['name'] for m in qwen_models]}")
        else:
            print("\n⚠️  No Qwen models found. You may need to pull it:")
            print("    ssh h100 'ollama pull qwen3:235b'")
            
    except json.JSONDecodeError:
        print(f"❌ Failed to parse response: {result.stdout[:200]}")
        return False
    
    # Test the generate endpoint
    print("\n3. Testing /api/generate endpoint...")
    test_data = json.dumps({
        "model": "qwen3:235b",
        "prompt": "Reply with just: test successful",
        "stream": False,
        "options": {"num_predict": 10}
    })
    
    result = subprocess.run(
        f"ssh h100 'curl -s -X POST http://localhost:11434/api/generate -d {repr(test_data)} -w \"\\nHTTP_STATUS:%{{http_code}}\"'",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if "HTTP_STATUS:404" in result.stdout:
        print("❌ /api/generate endpoint returned 404")
        print("   This might mean the model 'qwen3:235b' is not available")
    elif "HTTP_STATUS:200" in result.stdout:
        print("✅ /api/generate endpoint is working!")
    else:
        print(f"⚠️  Unexpected response: {result.stdout[:200]}")
    
    return True

if __name__ == "__main__":
    if check_ollama():
        print("\n✅ Ollama appears to be properly configured!")
    else:
        print("\n❌ There are issues with Ollama configuration.")
        sys.exit(1)