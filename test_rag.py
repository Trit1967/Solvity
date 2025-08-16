#!/usr/bin/env python3
"""
Test what's actually being extracted and searched
"""

from final_rag import ProRAG
from pathlib import Path
import sys

def test_document_processing(file_path):
    """Test document extraction and search"""
    
    print("="*60)
    print("📋 RAG DIAGNOSTIC TEST")
    print("="*60)
    
    rag = ProRAG()
    
    # Create a fake file object
    class FakeFile:
        def __init__(self, path):
            self.name = path
    
    # Test extraction
    print(f"\n1️⃣ Testing file: {file_path}")
    print("-"*40)
    
    filename = Path(file_path).name
    
    # Extract content based on type
    if filename.lower().endswith('.pdf'):
        print("📑 Detected as PDF")
        content = rag.extract_pdf_text(file_path)
    elif filename.lower().endswith('.docx'):
        print("📄 Detected as DOCX")
        content = rag.extract_docx_text(file_path)
    elif filename.lower().endswith('.txt'):
        print("📝 Detected as TXT")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    elif filename.lower().endswith('.md'):
        print("📋 Detected as Markdown")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    else:
        print(f"❓ Unknown type: {filename}")
        content = ""
    
    print(f"\n2️⃣ Extraction Result:")
    print("-"*40)
    print(f"Content length: {len(content)} characters")
    print(f"First 500 chars:\n{content[:500]}")
    
    if len(content) > 0:
        # Test chunking
        print(f"\n3️⃣ Testing Chunking:")
        print("-"*40)
        chunks = rag.smart_chunk(content, filename)
        print(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1} ({chunk['type']}):")
            print(f"Length: {len(chunk['text'])} chars")
            print(f"Preview: {chunk['text'][:150]}...")
        
        # Load it properly
        print(f"\n4️⃣ Loading Document:")
        print("-"*40)
        files = [FakeFile(file_path)]
        status, file_info, chunk_info = rag.load_documents(files)
        print(f"Status: {status}")
        print(f"Chunks: {chunk_info}")
        
        # Test semantic search
        print(f"\n5️⃣ Testing Semantic Search:")
        print("-"*40)
        
        test_queries = [
            "experience",
            "skills",
            "education",
            "work history",
            "bio",
            "summary"
        ]
        
        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            results = rag.semantic_search(query, top_k=3)
            if results:
                print(f"  Found {len(results)} matches:")
                for r in results[:2]:
                    print(f"    Score: {r['score']:.3f} | Preview: {r['text'][:100]}...")
            else:
                print("  ❌ No matches found")
    else:
        print("\n❌ No content extracted!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_document_processing(sys.argv[1])
    else:
        print("Usage: python test_rag.py <your_resume_file>")
        print("\nTesting with sample text...")
        
        # Create a test file
        test_file = "/tmp/test_resume.txt"
        with open(test_file, 'w') as f:
            f.write("""
John Doe
Software Engineer

EXPERIENCE:
- Senior Developer at Tech Corp (2020-2023)
- Led team of 5 engineers
- Built scalable microservices

EDUCATION:
- BS Computer Science, MIT (2016)

SKILLS:
- Python, JavaScript, Go
- AWS, Docker, Kubernetes
            """)
        
        test_document_processing(test_file)