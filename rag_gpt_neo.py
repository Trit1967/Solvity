#!/usr/bin/env python3
"""
RAG Chatbot using GPT-Neo (Runs 100% Locally)
No API keys needed, completely free
"""

import os
import torch
from typing import List, Optional
from pathlib import Path

# Core dependencies
from transformers import (
    GPTNeoForCausalLM, 
    GPT2Tokenizer,
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM
)
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    DirectoryLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class GPTNeoRAG:
    def __init__(self, 
                 model_size: str = "1.3B",
                 device: str = None,
                 vector_db_path: str = "./faiss_index"):
        """
        Initialize GPT-Neo RAG System
        
        Args:
            model_size: "125M", "1.3B", or "2.7B" (or "6B" for GPT-J)
            device: "cuda" or "cpu" (auto-detects if None)
            vector_db_path: Path to store vector database
        """
        self.vector_db_path = vector_db_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Initializing GPT-Neo RAG on {self.device}")
        
        # Model selection
        model_map = {
            "125M": "EleutherAI/gpt-neo-125M",
            "1.3B": "EleutherAI/gpt-neo-1.3B",
            "2.7B": "EleutherAI/gpt-neo-2.7B",
            "6B": "EleutherAI/gpt-j-6B"  # GPT-J
        }
        
        model_name = model_map.get(model_size, "EleutherAI/gpt-neo-1.3B")
        
        # Load tokenizer and model
        print(f"📥 Loading {model_name} (this may take a moment on first run)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with memory optimization
        if self.device == "cpu" or model_size in ["125M", "1.3B"]:
            # Load in full precision for CPU or smaller models
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                low_cpu_mem_usage=True
            )
        else:
            # Use 8-bit quantization for larger models on GPU
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            except ImportError:
                print("⚠️ Install bitsandbytes for 8-bit quantization: pip install bitsandbytes")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
        
        self.model.to(self.device)
        
        # Create text generation pipeline
        self.text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            do_sample=True,
            device=0 if self.device == "cuda" else -1
        )
        
        # Wrap in LangChain
        self.llm = HuggingFacePipeline(pipeline=self.text_pipeline)
        
        # Initialize embeddings (smaller model for efficiency)
        print("📥 Loading embedding model")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device}
        )
        
        # Load or create vector store
        self.vectorstore = None
        self.load_vectorstore()
        
        print("✅ GPT-Neo RAG Ready!")
    
    def load_vectorstore(self):
        """Load existing vector store if it exists"""
        if os.path.exists(self.vector_db_path):
            print(f"📂 Loading vector store from {self.vector_db_path}")
            self.vectorstore = FAISS.load_local(
                self.vector_db_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
    
    def ingest_documents(self, directory_path: str, chunk_size: int = 500):
        """
        Ingest documents and create vector store
        
        Args:
            directory_path: Path to documents directory
            chunk_size: Size of text chunks (smaller = more precise, larger = more context)
        """
        print(f"📄 Ingesting documents from {directory_path}")
        
        # Load different file types
        loaders = []
        
        # PDFs
        pdf_loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        loaders.append(pdf_loader)
        
        # Text files
        txt_loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        loaders.append(txt_loader)
        
        # Markdown files
        md_loader = DirectoryLoader(
            directory_path,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader
        )
        loaders.append(md_loader)
        
        # Load all documents
        all_documents = []
        for loader in loaders:
            try:
                docs = loader.load()
                all_documents.extend(docs)
                print(f"  Loaded {len(docs)} documents from {loader}")
            except Exception as e:
                print(f"  Error with loader: {e}")
        
        if not all_documents:
            print("❌ No documents found!")
            return
        
        print(f"📚 Total documents loaded: {len(all_documents)}")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(all_documents)
        print(f"🔪 Split into {len(chunks)} chunks")
        
        # Create vector store
        print("🔢 Creating embeddings (this may take a while)...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Save vector store
        self.vectorstore.save_local(self.vector_db_path)
        print(f"💾 Vector store saved to {self.vector_db_path}")
    
    def query(self, question: str, k: int = 3) -> str:
        """
        Query the RAG system
        
        Args:
            question: User's question
            k: Number of relevant chunks to retrieve
        """
        if self.vectorstore is None:
            return "❌ No documents ingested yet. Run ingest_documents() first."
        
        # Custom prompt for better responses
        prompt_template = """Use the following pieces of context to answer the question. 
If you don't know the answer based on the context, just say you don't know.

Context:
{context}

Question: {question}

Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        # Get response
        print("🤔 Thinking...")
        response = qa_chain({"query": question})
        
        # Extract answer and sources
        answer = response['result'].strip()
        sources = response.get('source_documents', [])
        
        # Add source references
        if sources:
            answer += "\n\n📚 Sources:"
            seen_sources = set()
            for doc in sources:
                source = Path(doc.metadata.get('source', 'Unknown')).name
                if source not in seen_sources:
                    answer += f"\n  • {source}"
                    seen_sources.add(source)
        
        return answer
    
    def chat(self):
        """Interactive chat interface"""
        print("\n" + "="*50)
        print("🤖 GPT-Neo RAG Chatbot")
        print("="*50)
        print("\nCommands:")
        print("  /ingest <path> - Ingest documents from directory")
        print("  /help         - Show this help")
        print("  /exit         - Quit")
        print("\n" + "="*50 + "\n")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    command = user_input.split()[0].lower()
                    
                    if command == "/exit":
                        print("👋 Goodbye!")
                        break
                    
                    elif command == "/help":
                        print("\nCommands:")
                        print("  /ingest <path> - Ingest documents")
                        print("  /help         - Show help")
                        print("  /exit         - Quit")
                    
                    elif command == "/ingest":
                        parts = user_input.split(maxsplit=1)
                        if len(parts) > 1:
                            self.ingest_documents(parts[1])
                        else:
                            print("Usage: /ingest <directory_path>")
                    
                    else:
                        print(f"Unknown command: {command}")
                    
                    continue
                
                # Regular query
                answer = self.query(user_input)
                print(f"\n🤖 Bot: {answer}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-Neo RAG Chatbot")
    parser.add_argument(
        "--model", 
        choices=["125M", "1.3B", "2.7B", "6B"],
        default="1.3B",
        help="Model size (default: 1.3B)"
    )
    parser.add_argument(
        "--ingest",
        type=str,
        help="Directory to ingest documents from"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device to run on (auto-detect if not specified)"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = GPTNeoRAG(
        model_size=args.model,
        device=args.device
    )
    
    # Ingest documents if specified
    if args.ingest:
        rag.ingest_documents(args.ingest)
    
    # Start chat
    rag.chat()

if __name__ == "__main__":
    main()