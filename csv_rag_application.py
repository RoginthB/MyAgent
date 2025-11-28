import streamlit as st
from csv_rag import CSVRAG
from pathlib import Path
import tempfile
import os
from typing import Generator

# Page configuration
st.set_page_config(
    page_title="CSV RAG Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat messages styling */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Success message styling */
    .element-container div[data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        padding: 20px 0;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'csv_rag' not in st.session_state:
    st.session_state.csv_rag = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'csv_indexed' not in st.session_state:
    st.session_state.csv_indexed = False
if 'current_csv' not in st.session_state:
    st.session_state.current_csv = None

def process_file(uploaded_file):
    """Process uploaded CSV or Excel file"""
    try:
        # Determine file extension
        suffix = Path(uploaded_file.name).suffix
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Initialize CSV RAG if not already done
        if st.session_state.csv_rag is None:
            st.session_state.csv_rag = CSVRAG()
        
        # Index the content
        with st.spinner('🔄 Indexing document content... This may take a moment.'):
            st.session_state.csv_rag.index_csv_content(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        st.session_state.csv_indexed = True
        st.session_state.current_csv = uploaded_file.name
        return True
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return False

def get_agent_response(query: str) -> tuple:
    """Get streaming response from agent with source information"""
    from langchain_community.tools import tool
    from langchain.agents import create_agent
    
    # Safety check for session state
    if 'csv_rag' not in st.session_state or st.session_state.csv_rag is None:
        yield ("⚠️ Error: CSV RAG system is not initialized. Please reload the page and upload a CSV.", [])
        return

    # Capture the instance locally to avoid session_state access issues inside the tool
    csv_rag_instance = st.session_state.csv_rag
    sources_info = []
    
    @tool(response_format="content_and_artifact")
    def retrieve_content(query):
        """Retrieve information to help answer a query from the CSV document."""
        # Retrieve more documents for comprehensive context
        # Use the captured local instance instead of st.session_state
        retrieve_docs = csv_rag_instance.vector_store.similarity_search_with_score(query=query, k=4)
        
        # Store source information for UI display
        for doc, score in retrieve_docs:
            page_num = doc.metadata.get('page', 'Unknown')
            sources_info.append({
                'page': page_num,
                'score': score,
                'relevance': f"{(1/(1+score))*100:.1f}%",
                'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        # Format with page numbers and relevance scores
        serialized_parts = []
        for doc, score in retrieve_docs:
            # CSVLoader puts row number in metadata 'row'
            row_num = doc.metadata.get('row', 'Unknown')
            source_info = f"[Row {row_num}, Relevance: {1/(1+score):.2f}]"
            serialized_parts.append(
                f"Source: {source_info}\n"
                f"Content: {doc.page_content}\n"
                f"{'='*80}"
            )
        
        serialized = "\n\n".join(serialized_parts)
        return serialized, [doc for doc, _ in retrieve_docs]
    
    system_prompt = """
    You are an expert assistant specialized in analyzing CSV documents and providing comprehensive, accurate answers.
    
    **Your Responsibilities:**
    1. Always use the retrieve_content tool to find relevant information from the CSV before answering
    2. Provide detailed, well-structured answers based on the retrieved context
    3. Cite specific row numbers when referencing information (e.g., "According to row 5...")
    4. If the retrieved content doesn't fully answer the question, acknowledge what you can and cannot answer
    5. Format your responses clearly with proper markdown formatting (headings, lists, bold text, etc.)
    6. When multiple sources provide information, synthesize them into a coherent response
    7. If you're uncertain about something, clearly state your level of confidence
    
    **Response Format:**
    - Start with a direct answer to the question
    - Provide supporting details and explanations
    - Include relevant examples or quotes from the document
    - End with source citations (row numbers)
    
    **Important:**
    - Be thorough but concise
    - Prioritize accuracy over speculation
    - Use the exact terminology from the document when appropriate
    - If information is contradictory, point it out
    """
    
    tools = [retrieve_content]
    
    # Create agent
    agent = create_agent(
        model=csv_rag_instance.model,
        tools=tools,
        system_prompt=system_prompt
    )
    
    # Stream response
    full_response = ""
    for chunk in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
        if chunk["messages"]:
            last_message = chunk["messages"][-1]
            if hasattr(last_message, 'content'):
                content = last_message.content
                
                # Normalize content if it's a list (e.g. from some models/tools)
                if isinstance(content, list):
                    text_content = ""
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_content += block.get("text", "")
                    content = text_content
                
                if content and content != full_response:
                    full_response = content
                    yield (content, sources_info)
if len(st.session_state.messages) == 0:

    # Header
    st.title("📊 CSV RAG Assistant")
    st.markdown("""
        <div class="info-box">
            <p style="margin: 0; font-size: 16px;">
                <strong>Welcome to CSV/Excel RAG Assistant!</strong><br>
                Upload a CSV or Excel document and ask questions about its content. 
                The AI will retrieve relevant information and provide intelligent answers.
            </p>
        </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel document to analyze"
    )
    
    if uploaded_file is not None:
        if st.session_state.current_csv != uploaded_file.name:
            st.session_state.csv_indexed = False
        
        if st.button("🚀 Process Document", use_container_width=True):
            if process_file(uploaded_file):
                st.success(f"✅ Successfully indexed: {uploaded_file.name}")
                st.balloons()
    
    # Display current document status
    st.markdown("---")
    st.header("📊 Status")
    
    if st.session_state.csv_indexed:
        st.success("✅ Document Indexed")
        st.info(f"📄 Current: {st.session_state.current_csv}")
    else:
        st.warning("⏳ No document indexed yet")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Info section
    st.markdown("---")
    st.markdown("""
        <div style="padding: 15px; background-color: rgba(255, 255, 255, 0.05); border-radius: 10px;">
            <h4 style="margin-top: 0;">💡 How to Use</h4>
            <ol style="font-size: 14px;">
                <li>Upload a CSV or Excel document</li>
                <li>Click "Process Document" to index</li>
                <li>Ask questions about the content</li>
                <li>Get AI-powered answers!</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# Main chat interface
if st.session_state.csv_indexed:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available (for assistant messages)
            # Display sources if available (for assistant messages)
            pass
    
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your CSV..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            sources_placeholder = st.empty()
            full_response = ""
            sources_data = []
            
            try:
                for response_chunk, sources in get_agent_response(prompt):
                    full_response = response_chunk
                    sources_data = sources
                    # message_placeholder.markdown(full_response + "▌")
                
                # Display final response
                message_placeholder.markdown(full_response)
                
                # Display sources in an expandable section
                # Display sources in an expandable section
                pass
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": sources_data
                })
            except Exception as e:
                error_message = f"❌ Error generating response: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    # Show welcome message
    st.markdown("""
        <div style="text-align: center; padding: 50px 20px;">
            <h2 style="color: #667eea;">👋 Getting Started</h2>
            <p style="font-size: 18px; color: #666;">
                Upload a CSV or Excel document using the sidebar to begin your intelligent document analysis journey.
            </p>
            <div style="margin-top: 30px; display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                            padding: 20px; border-radius: 15px; max-width: 250px;">
                    <h3 style="margin-top: 0;">🎯 Accurate</h3>
                    <p>Get precise answers based on actual document content</p>
                </div>
                <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                            padding: 20px; border-radius: 15px; max-width: 250px;">
                    <h3 style="margin-top: 0;">⚡ Fast</h3>
                    <p>Quick processing and instant responses</p>
                </div>
                <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                            padding: 20px; border-radius: 15px; max-width: 250px;">
                    <h3 style="margin-top: 0;">🤖 AI-Powered</h3>
                    <p>Advanced language models for intelligent analysis</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

if len(st.session_state.messages) == 0:
# Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 20px; color: #888;">
            <p style="margin: 0;">Built with ❤️ using Streamlit and LangChain</p>
            <p style="margin: 5px 0 0 0; font-size: 12px;">Powered by Google Gemini 2.5 Flash</p>
        </div>
    """, unsafe_allow_html=True)
