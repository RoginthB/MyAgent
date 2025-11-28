from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import tool
from langchain.agents import create_agent
from pathlib import Path
import streamlit as st
import pandas as pd
import tempfile
import os
import faiss

load_dotenv()

class CSVRAG:
    def __init__(self):
        self.model = init_chat_model("google_genai:gemini-2.5-flash")
        self.vector_store = None
        self.agent = None
        self.embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    def index_csv_content(self, file_path):
        # Handle Excel files
        if file_path.endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(file_path)
                # Create a temporary CSV file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_csv:
                    df.to_csv(tmp_csv.name, index=False)
                    csv_path = tmp_csv.name
            except Exception as e:
                raise ValueError(f"Error converting Excel to CSV: {e}")
        else:
            csv_path = file_path

        try:
            loader = CSVLoader(file_path=csv_path)
            docs = loader.load()
        finally:
            # Clean up temp CSV if we created one from Excel
            if file_path.endswith(('.xlsx', '.xls')) and 'csv_path' in locals():
                try:
                    os.unlink(csv_path)
                except Exception as e:
                    print(f"Error deleting temp CSV: {e}")

        # Splitting docs if needed, though CSVLoader usually does row-based loading
        # We can still use a splitter if rows are very long
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            add_start_index=True,
        )
        splitted_docs = text_splitter.split_documents(docs)
        
        if not splitted_docs:
            raise ValueError("No documents found to index.")

        embedding_dim = len(self.embedding.embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)
        self.vector_store = FAISS(
            embedding_function=self.embedding,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.vector_store.add_documents(splitted_docs)
        print("CSV documents stored in vector store")

    def query_agent(self, query: str):
        
        @tool(response_format="content_and_artifact")
        def retrieve_content(query):
            """Retrieve information to help answer a query from the CSV document."""
            # Retrieve documents
            if self.vector_store is None:
                return "Error: Vector store not initialized. Please upload and process a document first.", []
                
            retrieve_docs = self.vector_store.similarity_search_with_score(query=query, k=4)
            
            # Format with source info (row number usually comes in metadata for CSVLoader)
            serialized_parts = []
            for doc, score in retrieve_docs:
                # CSVLoader puts row number in metadata 'row'
                row_num = doc.metadata.get('row', 'Unknown')
                source_path = doc.metadata.get('source', 'Unknown')
                source_info = f"[Row {row_num}, Source: {Path(source_path).name}, Relevance: {1/(1+score):.2f}]"
                serialized_parts.append(
                    f"Source: {source_info}\n"
                    f"Content: {doc.page_content}\n"
                    f"{'='*80}"
                )
            
            serialized = "\n\n".join(serialized_parts)
            return serialized, [doc for doc, _ in retrieve_docs]
        
        system_prompt = """
        You are an expert assistant specialized in analyzing CSV data and providing comprehensive, accurate answers.
        
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
        - Include relevant examples or data points from the document
        - End with source citations (row numbers)
        
        **Important:**
        - Be thorough but concise
        - Prioritize accuracy over speculation
        - Use the exact terminology from the document when appropriate
        - If information is contradictory, point it out
        """
        tools = [retrieve_content]

        self.agent = create_agent(model=self.model, tools=tools, system_prompt=system_prompt)
        
        # Return the generator directly
        return self.agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values")

if __name__ == "__main__":
    # Test block
    csv_rag = CSVRAG()
    # Assuming a test file exists or will be created for testing
    # csv_path = Path("data/test.csv")
    # if csv_path.exists():
    #     csv_rag.index_csv_content(csv_path)
    #     for chunk in csv_rag.query_agent("What is the value?"):
    #         print(chunk)
