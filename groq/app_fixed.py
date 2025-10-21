import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Retrieve the Groq API key from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

# Path to store the vector embeddings
VECTOR_STORE_PATH = "vectors"
PDF_PATH = "D:\\iPsychiatrist\\groq\\New Oxford Textbook of Psychiatry-2161hlm.pdf"

@st.cache_resource
def get_embeddings():
    """Get embeddings model (cached)."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def initialize_vector_store():
    """Initialize or load vector store."""
    embeddings = get_embeddings()
    
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            # Try to load existing vector store
            loaded_vectors = FAISS.load_local(
                VECTOR_STORE_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            return loaded_vectors
        except Exception as e:
            st.warning(f"Could not load existing vectors: {e}. Creating new ones...")
            # If loading fails, delete old vectors and create new ones
            import shutil
            if os.path.exists(VECTOR_STORE_PATH):
                shutil.rmtree(VECTOR_STORE_PATH)
    
    # Create new vector store
    if not os.path.exists(PDF_PATH):
        st.error(f"PDF file not found at: {PDF_PATH}")
        st.stop()
    
    with st.spinner("Loading PDF and creating embeddings... This may take a few minutes."):
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        
        # Split documents into chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:50])
        
        # Create a vector store from the document chunks
        vectorstore = FAISS.from_documents(final_documents, embeddings)
        vectorstore.save_local(VECTOR_STORE_PATH)
        st.success("Embeddings successfully created and saved locally.")
        
    return vectorstore

# Custom CSS for styling the Streamlit app
st.markdown("""
    <style>
        body {
            background-image: url('https://media.istockphoto.com/id/1294477039/vector/metaphor-bipolar-disorder-mind-mental-double-face-split-personality-concept-mood-disorder-2.jpg?s=612x612&w=0&k=20&c=JtBxyFapXIA63hzZk_F5WNDF92J8fD2gIFNX3Ta4U3A=');
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
st.title("iPsychiatrist 🧠")

# Initialize vector store
try:
    loaded_vectors = initialize_vector_store()
except Exception as e:
    st.error(f"Error initializing vector store: {e}")
    st.stop()

# Initialize the language model
try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
except Exception as e:
    st.error(f"Error initializing ChatGroq: {e}")
    st.stop()

# Define the prompt template for the chat model
prompt_template = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
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
for user_prompt, answer in zip(st.session_state["user_prompt_history"], st.session_state["chat_answers_history"]):
    st.markdown(f"<div class='chat-message user'>{user_prompt}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-message assistant'>{answer}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Input for user prompt
prompt = st.text_input("Input your prompt here", key="user_input")

# Button to submit the prompt
if st.button("Submit Prompt"):
    if prompt:
        with st.spinner("Thinking..."):
            try:
                # Record the start time
                start = time.process_time()
                
                # Get the response from the retrieval chain
                response = retrieval_chain.invoke({"input": prompt})
                
                # Calculate response time
                response_time = time.process_time() - start
                
                # Display the user prompt and assistant response
                st.markdown(f"<div class='chat-message user'>{prompt}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-message assistant'>{response['answer']}</div>", unsafe_allow_html=True)
                
                # Display the response time
                st.caption(f"Response time: {response_time:.2f} seconds")

                # Update session state with the new chat history
                st.session_state["chat_answers_history"].append(response['answer'])
                st.session_state["user_prompt_history"].append(prompt)
                st.session_state["chat_history"].append((prompt, response['answer']))

                # Display the document similarity search results in an expander
                with st.expander("📄 Document Similarity Search"):
                    for i, doc in enumerate(response["context"]):
                        st.write(f"**Document {i+1}:**")
                        st.write(doc.page_content)
                        st.write("---")
                        
            except Exception as e:
                st.error(f"Error processing request: {e}")
    else:
        st.warning("Please enter a prompt before submitting.")
