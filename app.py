"""
AI Documentation Assistant
RAG-powered chatbot for searching and querying technical documentation

Author: Subramani Mokkala
"""

import streamlit as st
import os
from src.document_processor import process_document, get_document_stats
from src.embeddings import EmbeddingStore
from src.qa_engine import QAEngine

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="AI Documentation Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ===== SESSION STATE INITIALIZATION =====
if 'embedding_store' not in st.session_state:
    st.session_state['embedding_store'] = None
    st.session_state['qa_engine'] = None
    st.session_state['documents'] = []
    st.session_state['chat_history'] = []
    st.session_state['initialized'] = False

# ===== HELPER FUNCTIONS =====

def initialize_models():
    """Initialize embedding store and QA engine"""
    if not st.session_state['initialized']:
        with st.spinner("🔄 Initializing AI models (first time takes 1-2 minutes)..."):
            st.session_state['embedding_store'] = EmbeddingStore()
            st.session_state['qa_engine'] = QAEngine()
            st.session_state['initialized'] = True
        st.success("✅ Models initialized!")

def process_uploaded_files(uploaded_files):
    """Process uploaded documents and build index"""
    documents = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing {file.name}...")
        
        try:
            # Process document
            doc = process_document(file, file.name)
            documents.append(doc)
            
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    return documents

# ===== MAIN APP =====

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 AI Documentation Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Ask questions about your documents using AI-powered semantic search")
    st.markdown("---")
    
    # Initialize models
    initialize_models()
    
    # ===== SIDEBAR: Document Upload =====
    st.sidebar.header("📁 Document Management")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents",
        type=['pdf', 'txt', 'md'],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or Markdown files"
    )
    
    if uploaded_files and st.sidebar.button("🔨 Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            # Process files
            documents = process_uploaded_files(uploaded_files)
            
            if documents:
                # Build index
                st.session_state['embedding_store'].build_index(documents)
                st.session_state['documents'] = documents
                
                st.sidebar.success(f"✅ Processed {len(documents)} documents!")
                
                # Show stats
                st.sidebar.markdown("### 📊 Document Stats")
                for doc in documents:
                    stats = get_document_stats(doc)
                    st.sidebar.markdown(f"""
                    **{stats['filename']}**
                    - Chunks: {stats['number_of_chunks']}
                    - Characters: {stats['total_characters']:,}
                    """)
    
    st.sidebar.markdown("---")
    
    # Current status
    if st.session_state['documents']:
        st.sidebar.success(f"📚 {len(st.session_state['documents'])} documents loaded")
    else:
        st.sidebar.info("👆 Upload documents to get started")
    
    # ===== MAIN CONTENT =====
    
    # Show instructions if no documents
    if not st.session_state['documents']:
        st.info("""
        👋 **Welcome to AI Documentation Assistant!**
        
        **How to use:**
        1. Upload documents (PDF, TXT, MD) using the sidebar
        2. Click "Process Documents"
        3. Ask questions in natural language
        4. Get AI-powered answers with source citations
        
        **Features:**
        - 🔍 Semantic search (meaning-based, not just keywords)
        - 🤖 AI-powered question answering
        - 📝 Source attribution
        - 🚀 100% local processing (no API calls!)
        - 🔒 Your data never leaves your computer
        """)
        return
    
    # ===== CHAT INTERFACE =====
    
    st.header("💬 Ask Questions")
    
    # Display chat history
    if st.session_state['chat_history']:
        st.markdown("### 📜 Chat History")
        
        for i, chat in enumerate(st.session_state['chat_history']):
            # Question
            st.markdown(f"**🙋 You:** {chat['question']}")
            
            # Answer
            st.markdown(f"**🤖 Assistant:** {chat['answer']}")
            
            # Confidence and sources
            col1, col2 = st.columns([1, 3])
            with col1:
                confidence = chat.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.0%}")
            
            with col2:
                if chat.get('sources'):
                    with st.expander("📚 View Sources"):
                        for source in chat['sources']:
                            st.markdown(f"""
                            <div class="source-box">
                            <strong>📄 {source['document']}</strong><br>
                            Similarity: {source['similarity']:.2%}<br>
                            <em>{source['preview']}</em>
                            </div>
                            """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    # Question input
    question = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g., What is the main topic of these documents?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state['chat_history'] = []
            st.rerun()
    
    if ask_button and question:
        with st.spinner("🤔 Thinking..."):
            # Search for relevant contexts
            store = st.session_state['embedding_store']
            contexts = store.search(question, top_k=5)
            
            # Generate answer
            engine = st.session_state['qa_engine']
            result = engine.answer_question(question, contexts)
            
            # Add to history
            st.session_state['chat_history'].append({
                'question': question,
                'answer': result['answer'],
                'confidence': result['confidence'],
                'sources': result.get('sources', [])
            })
            
            # Rerun to show new message
            st.rerun()
    
    # ===== ADVANCED FEATURES =====
    
    with st.expander("⚙️ Advanced Options"):
        st.markdown("### Search Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_results = st.slider(
                "Number of search results",
                min_value=1,
                max_value=10,
                value=5,
                help="How many document chunks to retrieve"
            )
        
        with col2:
            show_context = st.checkbox(
                "Show retrieved context",
                value=False,
                help="Display the text chunks used to generate answers"
            )
        
        # Manual search (without Q&A)
        st.markdown("### 🔍 Manual Search")
        search_query = st.text_input("Search documents directly:", key="search_input")
        
        if st.button("Search", key="manual_search"):
            if search_query:
                store = st.session_state['embedding_store']
                results = store.search(search_query, top_k=num_results)
                
                st.markdown(f"**Found {len(results)} relevant chunks:**")
                
                for result in results:
                    st.markdown(f"""
                    <div class="source-box">
                    <strong>📄 {result['source_document']}</strong> 
                    (Similarity: {result['similarity_score']:.2%})<br><br>
                    {result['text']}
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()