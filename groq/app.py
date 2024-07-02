import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

load_dotenv()

# Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

VECTOR_STORE_PATH = "vectors"

def initialize_embeddings_and_vectors():
    if not os.path.exists(VECTOR_STORE_PATH):
        embeddings = HuggingFaceEmbeddings()
        loader = PyPDFLoader("D:\\iPsychiatrist\\groq\\New Oxford Textbook of Psychiatry-2161hlm.pdf")
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:50])
        
        vectorstore = FAISS.from_documents(final_documents, embeddings)
        vectorstore.save_local(VECTOR_STORE_PATH)
        print("Embeddings successfully created and saved locally.")
    else:
        print("Embeddings already exist. Loading from disk.")

# Initialize embeddings and vector store only once
initialize_embeddings_and_vectors()

# Load the saved vectors
embeddings = HuggingFaceEmbeddings()
loaded_vectors = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

st.markdown("""
    <style>
        body {
            background-image: url('https://www.example.com/mental_health_background.jpg'); /* Replace with a valid URL */
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: #333;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .chat-message {
            padding: 10px;
            margin: 10px 0;
            border-radius: 10px;
            background-color: #f0f0f0;
        }
        .chat-message.user {
            background-color: #d1e7dd;
        }
        .chat-message.assistant {
            background-color: #ffe5e5;
        }
        .stButton>button {
            color: #ffffff;
            background-color: #007bff;
            border-color: #007bff;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            border-radius: 5px;
        }
        .stTextInput>div>div>input {
            padding: 10px;
            font-size: 1rem;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        .stExpander>div>div {
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("iPsychiatrist")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

prompt_template = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
""")
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = loaded_vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

prompt = st.text_input("Input your prompt here")

if st.button("Submit Prompt"):
    if prompt:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        st.session_state["chat_answers_history"].append(response['answer'])
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_history"].append((prompt, response['answer']))

        # With a Streamlit expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

# Displaying the chat history
if st.session_state["chat_answers_history"]:
    for answer, user_prompt in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        message1 = st.chat_message("user")
        message1.write(user_prompt)
        message2 = st.chat_message("assistant")
        message2.write(answer)


