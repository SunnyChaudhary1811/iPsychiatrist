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

# Load environment variables from a .env file
load_dotenv()

# Retrieve the Groq API key from environment variables
groq_api_key = os.environ['GROQ_API_KEY']

# Path to store the vector embeddings
VECTOR_STORE_PATH = "vectors"

def initialize_embeddings_and_vectors():
    """Initializes embeddings and vector store if not already present."""
    if not os.path.exists(VECTOR_STORE_PATH):
        embeddings = HuggingFaceEmbeddings()
        loader = PyPDFLoader("D:\\iPsychiatrist\\groq\\New Oxford Textbook of Psychiatry-2161hlm.pdf")
        docs = loader.load()
        
        # Split documents into chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:50])
        
        # Create a vector store from the document chunks
        vectorstore = FAISS.from_documents(final_documents, embeddings)
        vectorstore.save_local(VECTOR_STORE_PATH)
        print("Embeddings successfully created and saved locally.")
    else:
        print("Embeddings already exist. Loading from disk.")

# Initialize embeddings and vector store only once
initialize_embeddings_and_vectors()

# Load the saved vector store
embeddings = HuggingFaceEmbeddings()
loaded_vectors = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# Custom CSS for styling the Streamlit app
st.markdown("""
    <style>
        body {
            background-image: url('https://media.istockphoto.com/id/1294477039/vector/metaphor-bipolar-disorder-mind-mental-double-face-split-personality-concept-mood-disorder-2.jpg?s=612x612&w=0&k=20&c=JtBxyFapXIA63hzZk_F5WNDF92J8fD2gIFNX3Ta4U3A='); /* Replace with a valid URL */
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: #333;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            margin: auto;
            margin-top: 40px;
        }
        .chat-message {
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 20px;
            background-color: #f0f0f0;
            max-width: 75%;
            word-wrap: break-word;
        }
        .chat-message.user {
            background-color: #d1e7dd;
            align-self: flex-end;
            margin-left: auto;
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
            border-radius: 20px;
            border: 1px solid #ccc;
            width: 100%;
            margin-top: 10px;
        }
        .stExpander>div>div {
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the Streamlit app
st.title("iPsychiatrist")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Define the prompt template for the chat model
prompt_template = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
""")

# Create the document chain for processing
document_chain = create_stuff_documents_chain(llm, prompt_template)

# Create a retriever from the loaded vector store
retriever = loaded_vectors.as_retriever()

# Create a retrieval chain using the retriever and document chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Initialize session state variables to store chat history
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display the chat container
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Display previous chat history
for answer, user_prompt in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
    st.markdown(f"<div class='chat-message user'>{user_prompt}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-message assistant'>{answer}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Input for user prompt
prompt = st.text_input("Input your prompt here")

# Button to submit the prompt
if st.button("Submit Prompt"):
    if prompt:
        # Record the start time
        start = time.process_time()
        
        # Get the response from the retrieval chain
        response = retrieval_chain.invoke({"input": prompt})
        
        # Display the response time
        st.write("Response time:", time.process_time() - start)
        
        # Display the user prompt and assistant response
        st.markdown(f"<div class='chat-message user'>{prompt}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-message assistant'>{response['answer']}</div>", unsafe_allow_html=True)

        # Update session state with the new chat history
        st.session_state["chat_answers_history"].append(response['answer'])
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_history"].append((prompt, response['answer']))

        # Display the document similarity search results in an expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
