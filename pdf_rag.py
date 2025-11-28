from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import tool
from langchain.agents import create_agent
from pathlib import Path
import streamlit as st
import faiss
load_dotenv()
 
class PDFRAG:
    def __init__(self):
        self.model = init_chat_model("google_genai:gemini-2.5-flash")
        self.vector_store = None
        self.agent =None
        self.embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    def index_pdf_content(self, pdf_path):
        pdf_loader = PyPDFLoader(file_path=pdf_path)
        docs = pdf_loader.load()

        #splitting docs with better chunking strategy
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for better context
            chunk_overlap=300,  # More overlap for continuity
            add_start_index=True,
        )
        splitted_docs = text_splitter.split_documents(docs)
        embedding_dim = len(self.embedding.embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)
        self.vector_store = FAISS(
            embedding_function=self.embedding,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.vector_store.add_documents(splitted_docs)
        print("document stored in vector store")

    def query_agent(self, query: str):
        
        @tool(response_format="content_and_artifact")
        def retrieve_content(query):
            """Retrieve information to help answer a query from the PDF document."""
            # Retrieve more documents for comprehensive context
            retrieve_docs = self.vector_store.similarity_search_with_score(query=query, k=4)
            
            # Format with page numbers and relevance scores
            serialized_parts = []
            for doc, score in retrieve_docs:
                page_num = doc.metadata.get('page', 'Unknown')
                source_info = f"[Page {page_num}, Relevance: {1/(1+score):.2f}]"
                serialized_parts.append(
                    f"Source: {source_info}\n"
                    f"Content: {doc.page_content}\n"
                    f"{'='*80}"
                )
            
            serialized = "\n\n".join(serialized_parts)
            return serialized, [doc for doc, _ in retrieve_docs]
        
        system_prompt = """
        You are an expert assistant specialized in analyzing PDF documents and providing comprehensive, accurate answers.
        
        **Your Responsibilities:**
        1. Always use the retrieve_content tool to find relevant information from the PDF before answering
        2. Provide detailed, well-structured answers based on the retrieved context
        3. Cite specific page numbers when referencing information (e.g., "According to page 5...")
        4. If the retrieved content doesn't fully answer the question, acknowledge what you can and cannot answer
        5. Format your responses clearly with proper markdown formatting (headings, lists, bold text, etc.)
        6. When multiple sources provide information, synthesize them into a coherent response
        7. If you're uncertain about something, clearly state your level of confidence
        
        **Response Format:**
        - Start with a direct answer to the question
        - Provide supporting details and explanations
        - Include relevant examples or quotes from the document
        - End with source citations (page numbers)
        
        **Important:**
        - Be thorough but concise
        - Prioritize accuracy over speculation
        - Use the exact terminology from the document when appropriate
        - If information is contradictory, point it out
        """
        tools = [retrieve_content]

        self.agent = create_agent(model=self.model, tools=tools, system_prompt=system_prompt)
        for chunk in self.agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
            chunk["messages"][-1].pretty_print()

if __name__ == "__main__":
    pdf_rag = PDFRAG()
    pdf_path = Path(__file__).parent /"data/Cricket_with_AI_Product_Development_Document.pdf"
    pdf_rag.index_pdf_content(pdf_path)
    pdf_rag.query_agent("What is the name of the book?")
