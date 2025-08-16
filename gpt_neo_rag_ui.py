#!/usr/bin/env python3
"""
GPT-Neo Powered RAG with Modern UI
Using EleutherAI's GPT-Neo for advanced Q&A
"""

import gradio as gr
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, pipeline
import torch
import os
from pathlib import Path
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

class GPTNeoRAG:
    def __init__(self, model_size="125M"):
        """Initialize GPT-Neo model"""
        print(f"\n🚀 Loading GPT-Neo {model_size}...")
        print("📥 First run will download the model\n")
        
        # Model selection
        models = {
            "125M": "EleutherAI/gpt-neo-125M",   # 500MB download, 2GB RAM
            "1.3B": "EleutherAI/gpt-neo-1.3B",   # 5GB download, 6GB RAM
            "2.7B": "EleutherAI/gpt-neo-2.7B"    # 10GB download, 12GB RAM
        }
        
        model_name = models.get(model_size, "EleutherAI/gpt-neo-125M")
        
        try:
            # Load tokenizer
            print(f"Loading tokenizer...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            print(f"Loading GPT-Neo {model_size} model...")
            self.model = GPTNeoForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            print(f"✅ GPT-Neo {model_size} loaded successfully!\n")
            self.model_loaded = True
            self.model_size = model_size
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            self.tokenizer = None
            self.model_loaded = False
        
        self.documents = {}
        self.chat_history = []
    
    def load_documents(self, files):
        """Load documents into memory"""
        if not files:
            return "No files uploaded", [], "No documents loaded"
        
        self.documents = {}
        loaded = []
        total_chars = 0
        
        for file in files:
            try:
                with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                filename = Path(file.name).name
                self.documents[filename] = content
                loaded.append(filename)
                total_chars += len(content)
                
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
        
        if loaded:
            status = f"✅ Loaded {len(loaded)} documents ({total_chars:,} characters)"
            file_info = [[f"📄 {f}", f"{len(self.documents[f]):,} chars"] for f in loaded]
            info = f"{len(loaded)} documents ready for Q&A"
            return status, file_info, info
        
        return "❌ No files loaded", [], "No documents loaded"
    
    def generate_answer(self, question: str, context: str, max_length: int = 150):
        """Generate answer using GPT-Neo"""
        if not self.model_loaded:
            return "Model not loaded. Please wait or restart."
        
        # Create a prompt for Q&A
        prompt = f"""Context: {context[:1500]}

Question: {question}

Answer: Based on the context above,"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                num_beams=3,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2
            )
        
        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
            # Clean up the answer
            answer = answer.replace("Based on the context above,", "").strip()
            # Stop at first period or newline
            if '.' in answer:
                answer = answer.split('.')[0] + '.'
            return answer
        
        return full_response[-max_length:]
    
    def answer_question(self, question: str, history: List[Tuple[str, str]]):
        """Process question and return answer"""
        
        if not self.documents:
            response = "📚 Please upload documents first in the Documents tab."
            return history + [(question, response)]
        
        if not question.strip():
            return history
        
        if not self.model_loaded:
            response = "⏳ Model is loading... Please wait a moment."
            return history + [(question, response)]
        
        # Find relevant context from documents
        question_lower = question.lower()
        relevant_chunks = []
        
        # Search for relevant content
        for filename, content in self.documents.items():
            # Split into paragraphs
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                # Simple relevance: does paragraph contain keywords from question?
                words = question_lower.split()
                if any(word in para.lower() for word in words if len(word) > 3):
                    relevant_chunks.append(para[:500])  # Limit paragraph size
                    if len(relevant_chunks) >= 3:  # Max 3 relevant chunks
                        break
        
        if not relevant_chunks:
            # Use first part of all documents as context
            context = " ".join([doc[:500] for doc in self.documents.values()])
        else:
            context = " ".join(relevant_chunks)
        
        try:
            # Generate answer
            answer = self.generate_answer(question, context)
            
            # Format response
            response = f"💡 {answer}"
            
            # Add source info if we can identify it
            for filename, content in self.documents.items():
                if any(chunk in content for chunk in relevant_chunks[:1]):
                    response += f"\n\n📄 *Source: {filename}*"
                    break
                    
        except Exception as e:
            print(f"Error generating answer: {e}")
            response = "❌ Error processing question. Try a simpler query."
        
        return history + [(question, response)]
    
    def clear_chat(self):
        """Clear chat history"""
        self.chat_history = []
        return []
    
    def get_model_info(self):
        """Get model information"""
        if self.model_loaded:
            return f"✅ GPT-Neo {self.model_size} loaded and ready"
        return "⏳ Model loading..."

def create_ui(model_size="125M"):
    """Create the Gradio UI"""
    
    rag = GPTNeoRAG(model_size=model_size)
    
    # Custom CSS for beautiful UI
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .gr-button-primary {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .gr-button-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2) !important;
    }
    .container {
        border-radius: 20px !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1) !important;
        background: white !important;
        padding: 2rem !important;
    }
    #chatbot {
        border-radius: 15px !important;
        border: 1px solid #e0e0e0 !important;
    }
    .user-row {
        background: linear-gradient(90deg, #e3f2fd 0%, #f3e5f5 100%) !important;
        border-radius: 15px !important;
        padding: 10px !important;
    }
    .bot-row {
        background: #f5f5f5 !important;
        border-radius: 15px !important;
        padding: 10px !important;
    }
    """
    
    with gr.Blocks(
        title="GPT-Neo RAG Assistant",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="indigo"
        ),
        css=custom_css
    ) as app:
        
        # Header with gradient
        gr.HTML("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0; font-size: 2.5rem;">🤖 GPT-Neo RAG Assistant</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem;">Powered by EleutherAI's Open-Source GPT-Neo</p>
        </div>
        """)
        
        # Model status
        with gr.Row():
            model_status = gr.Textbox(
                value=rag.get_model_info(),
                label="Model Status",
                interactive=False
            )
        
        # Main tabs
        with gr.Tabs():
            
            # Chat Tab
            with gr.Tab("💬 Chat with AI"):
                chatbot = gr.Chatbot(
                    height=450,
                    elem_id="chatbot"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask me anything about your documents...",
                        label="Your Question",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat")
                    
                    with gr.Column():
                        gr.Examples(
                            examples=[
                                "What is the main topic?",
                                "Summarize this document",
                                "What are the key points?",
                                "Explain the conclusion"
                            ],
                            inputs=msg,
                            label="Example Questions"
                        )
            
            # Documents Tab
            with gr.Tab("📁 Documents"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_upload = gr.File(
                            label="Upload Your Documents",
                            file_count="multiple",
                            file_types=[".txt", ".md", ".log", ".csv", ".json"],
                            height=250
                        )
                        
                        upload_btn = gr.Button(
                            "📥 Load Documents", 
                            variant="primary", 
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        upload_status = gr.Textbox(
                            label="Upload Status",
                            interactive=False
                        )
                        
                        file_list = gr.Dataframe(
                            headers=["Document", "Size"],
                            label="Loaded Documents",
                            interactive=False
                        )
                        
                        doc_info = gr.Textbox(
                            label="Documents Info",
                            interactive=False
                        )
            
            # About Tab
            with gr.Tab("ℹ️ About"):
                gr.Markdown(f"""
                ## About GPT-Neo RAG
                
                This system uses **GPT-Neo {model_size}** from EleutherAI:
                
                ### Model Specifications
                - **Type**: Open-source GPT model
                - **Size**: {model_size} parameters
                - **Creator**: EleutherAI
                - **License**: MIT (completely free)
                - **Privacy**: 100% local, no data leaves your computer
                
                ### Available Models
                | Model | Parameters | RAM Needed | Quality |
                |-------|-----------|------------|---------|
                | GPT-Neo-125M | 125 million | 2GB | Good for basic Q&A |
                | GPT-Neo-1.3B | 1.3 billion | 6GB | Better understanding |
                | GPT-Neo-2.7B | 2.7 billion | 12GB | Best quality |
                
                ### How It Works
                1. **Upload** your documents (text, markdown, logs, etc.)
                2. **Ask** questions about the content
                3. **GPT-Neo** analyzes and generates intelligent answers
                4. All processing happens **locally** on your machine
                
                ### Tips for Best Results
                - Upload related documents for better context
                - Ask specific questions
                - The AI will search through your documents to find answers
                - Longer documents may take a moment to process
                
                ---
                *Built with ❤️ using open-source AI*
                """)
        
        # Event handlers
        def send_message(msg, history):
            return "", rag.answer_question(msg, history)
        
        msg.submit(send_message, [msg, chatbot], [msg, chatbot])
        send_btn.click(send_message, [msg, chatbot], [msg, chatbot])
        
        upload_btn.click(
            rag.load_documents,
            inputs=[file_upload],
            outputs=[upload_status, file_list, doc_info]
        )
        
        clear_btn.click(
            rag.clear_chat,
            outputs=[chatbot]
        )
        
        # Auto-refresh model status
        app.load(
            rag.get_model_info,
            outputs=[model_status]
        )
    
    return app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["125M", "1.3B", "2.7B"],
        default="125M",
        help="GPT-Neo model size"
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 Starting GPT-Neo RAG Assistant")
    print("="*60)
    print(f"Model: GPT-Neo {args.model}")
    print(f"RAM needed: {'2GB' if args.model=='125M' else '6GB' if args.model=='1.3B' else '12GB'}")
    print("="*60 + "\n")
    
    app = create_ui(model_size=args.model)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False
    )