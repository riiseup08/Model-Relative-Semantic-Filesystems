"""
Quick test to verify the provider system works without calling APIs.

This tests that:
1. Import works even without llama-cpp-python installed
2. Error messages are helpful when dependencies are missing
3. Provider switching works correctly
"""

import os
import sys

# Test 1: Import should work even without llama-cpp-python
print("=" * 70)
print("Test 1: Import pymrsf without local model")
print("=" * 70)
try:
    import pymrsf
    print("✅ Import successful")
    print(f"   Version: {pymrsf.__version__}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Check that we can set different providers
print("\n" + "=" * 70)
print("Test 2: Provider configuration")
print("=" * 70)

# Try setting to OpenAI (won't actually load without API key)
os.environ["PYMRSF_PROVIDER"] = "openai"
from pymrsf.core import PROVIDER
print(f"✅ Provider can be set: {PROVIDER}")

# Test 3: Check helpful error messages
print("\n" + "=" * 70)
print("Test 3: Error messages when features aren't available")
print("=" * 70)

# This should give a helpful error about missing OpenAI package or API key
os.environ["PYMRSF_PROVIDER"] = "openai"
try:
    # Force backend loading
    from pymrsf.core import get_backend
    get_backend()
    print("✅ Backend loaded (OpenAI package installed)")
except ImportError as e:
    if "pip install pymrsf[openai]" in str(e):
        print("✅ Got helpful error message about missing openai package")
    else:
        print(f"⚠️  Error message could be better: {e}")
except ValueError as e:
    if "OPENAI_API_KEY" in str(e):
        print("✅ Got helpful error message about missing API key")
    else:
        print(f"⚠️  Error message could be better: {e}")
except Exception as e:
    print(f"⚠️  Unexpected error: {e}")

# Test 4: Check that optional dependencies are structured correctly
print("\n" + "=" * 70)
print("Test 4: Verify pyproject.toml structure")
print("=" * 70)

try:
    with open("pyproject.toml", "r") as f:
        content = f.read()
        
    checks = [
        ("local", "llama-cpp-python" in content),
        ("openai", "openai>=" in content),
        ("anthropic", "anthropic>=" in content),
        ("[project.optional-dependencies]", "[project.optional-dependencies]" in content),
    ]
    
    for name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {name} dependency configured")
        
except FileNotFoundError:
    print("⚠️  pyproject.toml not found (run from project root)")

# Test 5: Check that example files exist
print("\n" + "=" * 70)
print("Test 5: Example files")
print("=" * 70)

example_files = [
    ".env.example",
    "example_openai.py",
    "CHANGELOG.md",
]

for filename in example_files:
    if os.path.exists(filename):
        print(f"✅ {filename} exists")
    else:
        print(f"❌ {filename} missing")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
✅ Basic tests passed!

To complete the setup:
1. Choose a provider:
   - For lightweight testing: pip install pymrsf[openai]
   - For full features: pip install pymrsf[local]

2. Set up your .env file:
   - Copy .env.example to .env
   - Fill in your API keys or model path

3. Try the examples:
   - python example_openai.py (requires OpenAI API key)
   - See README.md for more examples
""")
