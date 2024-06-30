import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time
import pickle

# Load environment variables
load_dotenv()

# Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

# Function to initialize session state
def initialize_session():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFLoader("New Oxford Textbook of Psychiatry-2161hlm.pdf")
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        
        # Save vectors to a file if not already saved
        if os.path.exists("vectors.pkl"):
            with open("vectors.pkl", "rb") as f:
                st.session_state.vectors = pickle.load(f)
        else:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            with open("vectors.pkl", "wb") as f:
                pickle.dump(st.session_state.vectors, f)

initialize_session()

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Title
st.markdown("<h1 class='main-title'>iPsychiatrist</h1>", unsafe_allow_html=True)

# LLM Configuration
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Prompt Template
prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User Input
prompt_input = st.text_input("Input your prompt here", key="prompt_input", help="Type your question related to psychiatry here.")

# Process User Input
if prompt_input:
    start_time = time.time()
    response = retrieval_chain.invoke({"input": prompt_input})
    response_time = time.time() - start_time
    print("Response time:", response_time)

    st.markdown("<div class='response-container'>", unsafe_allow_html=True)
    st.write(response['answer'])
    st.markdown("</div>", unsafe_allow_html=True)

    # With a Streamlit expander
    with st.expander("Document Similarity Search", expanded=True):
        st.markdown("<div class='expander'>", unsafe_allow_html=True)
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
        st.markdown("</div>", unsafe_allow_html=True)

