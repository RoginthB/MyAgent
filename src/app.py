import streamlit as st
from web_RAG import WebRAG

# Page config
st.set_page_config(
    page_title="Web RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "web_rag" not in st.session_state:
    st.session_state.web_rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "current_url" not in st.session_state:
    st.session_state.current_url = ""

# Sidebar
with st.sidebar:
    st.title("Web RAG Assistant")
    st.markdown("---")
    
    st.subheader("🌐 Index Web Content")
    web_path = st.text_input(
        "Enter URL to index:",
        value="https://www.theblogstarter.com/step-1-get-started/",
        placeholder="https://example.com/article"
    )
    
    if st.button("🔍 Index Website", use_container_width=True):
        if web_path:
            with st.spinner("📥 Loading and indexing content..."):
                try:
                    # Initialize WebRAG if not already done
                    if st.session_state.web_rag is None:
                        st.session_state.web_rag = WebRAG()
                    
                    # Index the content
                    st.session_state.web_rag.index_web_content(web_path)
                    st.session_state.indexed = True
                    st.session_state.current_url = web_path
                    st.session_state.messages = []  # Clear previous chat
                    
                    st.success("✅ Content indexed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error indexing content: {str(e)}")
        else:
            st.warning("⚠️ Please enter a URL")
    
    st.markdown("---")
    
    # Display status
    if st.session_state.indexed:
        st.success("✅ Ready to answer questions")
        st.caption(f"📄 Indexed: {st.session_state.current_url[:50]}...")
    else:
        st.info("ℹ️ Please index a website first")
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# Main content area
st.title("💬 Chat with Web Content")

if not st.session_state.indexed:
    st.info("👈 Please index a website using the sidebar to start chatting!")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the web content..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response with streaming
        with st.chat_message("assistant"):
            try:
                # Use st.write_stream for real-time streaming display
                response = st.write_stream(st.session_state.web_rag.query_agent_stream(prompt))
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
