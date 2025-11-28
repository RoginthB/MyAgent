
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.agents import create_agent
from langchain_community.tools import tool
import streamlit as st

load_dotenv()

class WebRAG:
    
    def __init__(self):
        self.vector_store = None
        self.agent = None
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.model = init_chat_model("google_genai:gemini-2.5-flash")

    def index_web_content(self, web_path):

        embedding_dim = len(self.embeddings.embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)
        
        self.vector_store =FAISS(
        embedding_function=self.embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        )
        
        #loading web content
        loader = WebBaseLoader(
        web_path=web_path,
        )
        docs = loader.load()

        assert len(docs)==1
        print(f"Content length: {len(docs[0].page_content)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        splitted_docs = text_splitter.split_documents(docs)
        #storing documents

        document_ids = self.vector_store.add_documents(documents=splitted_docs)
        # Create agent after indexing
        
        self._create_agent()

    def _create_agent(self):
        """Create the agent with tools. Called internally after indexing."""
        # Create the tool with access to self
        @tool(response_format="content_and_artifact")
        def retrieve_content(query: str):
            """Retrieve information to help answer a query."""
            retrieve_docs = self.vector_store.similarity_search(query=query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieve_docs
            )
            return serialized, retrieve_docs

        tools = [retrieve_content]

        system_prompt = ("You have access to a tool that can retrieve context from a blog page. "
                        "Use this tool whenever you need to answer a question.")

        self.agent = create_agent(model=self.model, tools=tools, system_prompt=system_prompt)

    def query_agent(self, query: str):
        """Query the agent and return the response. For Streamlit integration."""
        if self.agent is None:
            raise ValueError("Agent not initialized. Call index_web_content first.")
        
        all_messages = []
        for step in self.agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
            all_messages = step["messages"]
        
        # Find the last AI message that has text content (not just tool calls)
        for msg in reversed(all_messages):
            # Check if it's an AI message with actual content
            if hasattr(msg, 'type') and msg.type == 'ai':
                content = msg.content if hasattr(msg, 'content') else None
                
                # Handle different content types
                if content:
                    # If content is a string
                    if isinstance(content, str) and content.strip():
                        return content
                    # If content is a list, extract text from it
                    elif isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                text_parts.append(item['text'])
                            elif isinstance(item, str):
                                text_parts.append(item)
                        
                        result = ' '.join(text_parts).strip()
                        if result:
                            return result
        
        return "No response generated."
    
    def query_agent_stream(self, query: str):
        """Query the agent and stream the response in real-time. Yields content as it's generated."""
        if self.agent is None:
            raise ValueError("Agent not initialized. Call index_web_content first.")
        
        # Track the last content we've seen to yield only new content
        last_content = ""
        
        for step in self.agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
            messages = step["messages"]
            
            # Look for the last AI message
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == 'ai':
                    content = msg.content if hasattr(msg, 'content') else None
                    
                    if content:
                        current_text = ""
                        
                        # Extract text from content
                        if isinstance(content, str):
                            current_text = content
                        elif isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and 'text' in item:
                                    text_parts.append(item['text'])
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            current_text = ' '.join(text_parts)
                        
                        # Yield only new content
                        if current_text and current_text != last_content:
                            # Yield the new portion
                            new_content = current_text[len(last_content):]
                            if new_content:
                                yield new_content
                            last_content = current_text
                    
                    break  # Only process the last AI message

    def web_rag_agent(self, web_path):
        """Console-based RAG agent (for backward compatibility)."""
        self.index_web_content(web_path)
        self._create_agent()

        input_query = input("Enter your query: ")

        for step in self.agent.stream({"messages": [{"role": "user", "content": input_query}]}, stream_mode="values"):
            step["messages"][-1].pretty_print()


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
