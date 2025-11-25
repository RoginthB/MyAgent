import streamlit as st
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

load_dotenv()

st.title("Web RAG")
model = init_chat_model("google_genai:gemini-2.5-flash")

class WebRAG:
    def __init__(self):
        self.vector_store = None
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

        print(f"Split blog post into {len(splitted_docs)} sub-documents")

        #storing documents

        document_ids = self.vector_store.add_documents(documents=splitted_docs)
        print(document_ids[:3])

    @tool(response_format ="content_and_artifact")
    def retrieve_content(query:str):
        """Retrieve information to help answer a query."""
        retrieve_docs = self.vector_store.similarity_search(query=query, k=2)
        serialized ="\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")for doc in retrieve_docs
        )
        return serialized, retrieve_docs

    def web_rag_agent(self, web_path):

        self.index_web_content(web_path)

        tools =[self.retrieve_content]

        system_prompt = ("You have a access to a tools that can retrieve context from a blog page." " Use this tool whenver you need to answer a question.")

        agent = create_agent(model=self.model, tools = tools, system_prompt=system_prompt)

        input_query =input("Enter your query: ")

        for step in agent.stream({"messages":[{"role":"user", "content":input_query}]}, stream_mode="values"):
            step["messages"][-1].pretty_print()


