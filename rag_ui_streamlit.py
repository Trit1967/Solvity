#!/usr/bin/env python3
"""
Streamlit UI for GPT-Neo RAG Chatbot
Modern, responsive interface with real-time updates
"""

import streamlit as st
import os
import torch
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import json

# Import our RAG system
from rag_gpt_neo import GPTNeoRAG

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    
if 'documents_ingested' not in st.session_state:
    st.session_state.documents_ingested = False
    
if 'ingested_files' not in st.session_state:
    st.session_state.ingested_files = []

def load_model(model_size):
    """Load the GPT-Neo model"""
    with st.spinner(f"Loading {model_size} model... This may take a minute..."):
        try:
            st.session_state.rag_system = GPTNeoRAG(model_size=model_size)
            st.session_state.model_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

def ingest_documents(uploaded_files):
    """Ingest uploaded documents"""
    if not uploaded_files:
        st.warning("No files uploaded")
        return False
    
    if st.session_state.rag_system is None:
        st.error("Please load a model first")
        return False
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        file_names = []
        
        # Save uploaded files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_names.append(uploaded_file.name)
        
        # Ingest documents
        with st.spinner(f"Processing {len(uploaded_files)} files..."):
            try:
                st.session_state.rag_system.ingest_documents(temp_dir)
                st.session_state.documents_ingested = True
                st.session_state.ingested_files.extend(file_names)
                return True
            except Exception as e:
                st.error(f"Error ingesting documents: {str(e)}")
                return False

def get_response(question):
    """Get response from RAG system"""
    if st.session_state.rag_system is None:
        return "❌ Please load a model first"
    
    if not st.session_state.documents_ingested:
        return "❌ Please ingest some documents first"
    
    try:
        response = st.session_state.rag_system.query(question)
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Main UI
def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🤖 GPT-Neo RAG Chatbot")
        st.caption("Local, Private, and Free Document Q&A")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("Model Settings")
        model_size = st.selectbox(
            "Select Model Size",
            options=["125M", "1.3B", "2.7B", "6B"],
            index=1,
            help="Larger models are more accurate but slower"
        )
        
        if st.button("🔄 Load Model", use_container_width=True):
            if load_model(model_size):
                st.success(f"✅ {model_size} model loaded!")
                st.rerun()
        
        # Model info
        if st.session_state.model_loaded:
            st.info(f"📊 Model: {model_size}")
            device = "GPU" if torch.cuda.is_available() else "CPU"
            st.info(f"💻 Device: {device}")
        
        st.divider()
        
        # Document upload
        st.subheader("📄 Document Management")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            help="Select documents to add to the knowledge base"
        )
        
        if st.button("📥 Ingest Documents", use_container_width=True):
            if ingest_documents(uploaded_files):
                st.success(f"✅ Ingested {len(uploaded_files)} files!")
                st.rerun()
        
        # Show ingested files
        if st.session_state.ingested_files:
            st.subheader("📚 Ingested Files")
            for file in st.session_state.ingested_files:
                st.text(f"• {file}")
        
        st.divider()
        
        # Info section
        st.subheader("ℹ️ About")
        st.markdown("""
        **100% Local & Private**
        - No cloud APIs
        - No data sharing
        - Works offline
        
        **Model Sizes:**
        - 125M: 2GB RAM (fast)
        - 1.3B: 6GB RAM (balanced)
        - 2.7B: 12GB RAM (better)
        - 6B: 16GB+ RAM (best)
        """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📈 Analytics", "📖 Help"])
    
    with tab1:
        # Chat interface
        if not st.session_state.model_loaded:
            st.warning("👈 Please load a model from the sidebar to start")
        elif not st.session_state.documents_ingested:
            st.info("👈 Please upload and ingest some documents to begin")
        else:
            # Chat history display
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.write(message["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.write(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your documents..."):
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Get bot response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = get_response(prompt)
                    st.write(response)
                
                # Add bot response to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Clear chat button
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
    
    with tab2:
        st.subheader("📊 Session Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Messages", len(st.session_state.chat_history))
        
        with col2:
            st.metric("Documents Ingested", len(st.session_state.ingested_files))
        
        with col3:
            status = "Active" if st.session_state.model_loaded else "Not Loaded"
            st.metric("Model Status", status)
        
        # Chat history stats
        if st.session_state.chat_history:
            st.subheader("📝 Conversation History")
            
            # Export chat button
            chat_json = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                label="💾 Download Chat History",
                data=chat_json,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Display recent questions
            st.subheader("Recent Questions")
            user_messages = [m for m in st.session_state.chat_history if m["role"] == "user"]
            for i, msg in enumerate(user_messages[-5:], 1):
                st.text(f"{i}. {msg['content'][:100]}...")
    
    with tab3:
        st.subheader("📖 How to Use")
        
        st.markdown("""
        ### Getting Started
        
        1. **Load a Model** 
           - Choose a model size based on your RAM
           - Click "Load Model" in the sidebar
        
        2. **Upload Documents**
           - Select PDF, TXT, MD, or DOCX files
           - Click "Ingest Documents" to process them
        
        3. **Ask Questions**
           - Type questions in the chat box
           - The AI will search your documents and provide answers
        
        ### Tips for Best Results
        
        - 🎯 Ask specific questions
        - 📝 Reference document topics
        - 🔍 Use keywords from your documents
        - ⏳ Be patient with large documents
        
        ### Troubleshooting
        
        **Model won't load?**
        - Check you have enough RAM
        - Try a smaller model size
        
        **Slow responses?**
        - Use a smaller model
        - Reduce document size
        - Consider using GPU
        
        **Out of memory?**
        - Restart the app
        - Use smaller model
        - Process fewer documents at once
        """)

if __name__ == "__main__":
    main()