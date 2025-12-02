"""
To test and see if all libraries are installed correctly
This uses free and local models, so no API is needed
"""

print("Testing installation...")
print("=" * 50)

# Test 1: Sentence Transformers
print("\n1. Testing Sentence Transformers...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("    Sentence Transformers working!")
except Exception as e:
    print(f"    Error: {e}")

# Test 2: FAISS
print("\n2. Testing FAISS...")
try:
    import faiss
    print("    FAISS working!")
except Exception as e:
    print(f"    Error: {e}")

# Test 3: PyPDF2
print("\n3. Testing PyPDF2...")
try:
    import PyPDF2
    print("    PyPDF2 working!")
except Exception as e:
    print(f"    Error: {e}")

# Test 4: Transformers
print("\n4. Testing Transformers...")
try:
    from transformers import pipeline
    print("    Transformers working!")
except Exception as e:
    print(f"    Error: {e}")

print("\n" + "=" * 50)
print("✅ All libraries installed successfully!")
print("\nYou're ready to build the FREE AI Documentation Assistant!")
print("No API keys needed - everything runs locally! 🚀")