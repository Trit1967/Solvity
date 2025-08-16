#!/usr/bin/env python3
"""
Simple RAG Chatbot using Open-Source Components
Cost: $0 (runs locally)
"""

import os
from typing import List, Dict
from pathlib import Path

# Core dependencies
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class RAGChatbot:
    def __init__(self, 
                 model_name: str = "llama3.2",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize RAG Chatbot with open-source components
        
        Args:
            model_name: Ollama model name (llama3.2, mistral, phi3, etc.)
            embedding_model: HuggingFace embedding model
            persist_directory: Directory to store vector database
        """
        self.persist_directory = persist_directory
        
        # Initialize embeddings (runs locally, no API needed)
        print(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize LLM (requires Ollama installed)
        print(f"Initializing LLM: {model_name}")
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = Ollama(
            model=model_name,
            callback_manager=callback_manager,
            temperature=0.7
        )
        
        # Initialize or load vector store
        self.vectorstore = None
        self.load_vectorstore()
        
    def load_vectorstore(self):
        """Load existing vector store or create new one"""
        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector store from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("No existing vector store found. Please ingest documents first.")
    
    def ingest_documents(self, directory_path: str, file_patterns: List[str] = None):
        """
        Ingest documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            file_patterns: List of file patterns to load (e.g., ['*.pdf', '*.txt'])
        """
        if file_patterns is None:
            file_patterns = ['*.pdf', '*.txt', '*.md', '*.docx']
        
        documents = []
        
        # Load documents
        for pattern in file_patterns:
            if pattern == '*.pdf':
                loader = DirectoryLoader(
                    directory_path,
                    glob=pattern,
                    loader_cls=PyPDFLoader
                )
            else:
                loader = DirectoryLoader(
                    directory_path,
                    glob=pattern,
                    loader_cls=TextLoader
                )
            
            try:
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded {len(docs)} documents matching {pattern}")
            except Exception as e:
                print(f"Error loading {pattern}: {e}")
        
        if not documents:
            print("No documents found to ingest")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Create or update vector store
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(chunks)
        
        # Persist the vector store
        self.vectorstore.persist()
        print(f"Vector store saved to {self.persist_directory}")
        
    def query(self, question: str, k: int = 4) -> str:
        """
        Query the RAG system
        
        Args:
            question: User's question
            k: Number of relevant chunks to retrieve
            
        Returns:
            Answer from the LLM
        """
        if self.vectorstore is None:
            return "No documents have been ingested yet. Please run ingest_documents() first."
        
        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True
        )
        
        # Get response
        response = qa_chain({"query": question})
        
        # Format response with sources
        answer = response['result']
        sources = response.get('source_documents', [])
        
        if sources:
            answer += "\n\nSources:"
            for i, doc in enumerate(sources[:3], 1):
                source = doc.metadata.get('source', 'Unknown')
                answer += f"\n{i}. {Path(source).name}"
        
        return answer
    
    def chat(self):
        """Interactive chat interface"""
        print("\n🤖 RAG Chatbot Ready!")
        print("Type 'exit' to quit, 'help' for commands\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  exit - Quit the chatbot")
                    print("  help - Show this help message")
                    print("  ingest <path> - Ingest documents from directory")
                    continue
                elif user_input.lower().startswith('ingest'):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        self.ingest_documents(parts[1])
                    else:
                        print("Usage: ingest <directory_path>")
                    continue
                
                print("\nBot: ", end="")
                answer = self.query(user_input)
                if not answer.startswith("Bot:"):  # Streaming already printed
                    print(answer)
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

def main():
    """Example usage"""
    # Initialize chatbot
    bot = RAGChatbot(
        model_name="llama3.2",  # or "mistral", "phi3", etc.
        embedding_model="all-MiniLM-L6-v2"
    )
    
    # Example: Ingest documents (uncomment to use)
    # bot.ingest_documents("./documents")
    
    # Start interactive chat
    bot.chat()

if __name__ == "__main__":
    main()