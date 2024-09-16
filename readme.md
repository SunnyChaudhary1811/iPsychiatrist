1. User Input (Prompt Submission)
Description:

 The user interacts with the chatbot through the Streamlit interface by submitting a question or query in the text input box.

 
Components:


Streamlit Frontend:

The app interface allows users to submit prompts. When the user types a question, it gets captured by the app.
st.text_input("Input your prompt here"): This takes the input from the user.


3. Prompt Capture and Processing

   
Description:

The input prompt is processed after the user clicks the "Submit Prompt" button.


Components:


if st.button("Submit Prompt"):: Captures the prompt input when the user clicks the button.
Prompt stored in session state: Ensures the prompt is saved for further processing and historical display.


5. Document Retrieval (Retriever)

   
Description:

The chatbot retrieves relevant document chunks based on the user’s query.


Components:


FAISS (Facebook AI Similarity Search): This is a vector-based retrieval system that compares the user query to vector embeddings of the document chunks (derived from a PDF in this case).

Vector Embeddings:

The embeddings are generated using HuggingFace models and represent the PDF chunks as numerical vectors.
retriever = loaded_vectors.as_retriever(): Converts the FAISS vectors into a retriever that can search for relevant chunks.


Flow:


The system compares the prompt to the embedded document chunks and retrieves the most relevant sections.


7. Context Generation

   
Description: After retrieving the document chunks, they are combined to form the context needed for the language model to generate an accurate response.


Components:


text_splitter = RecursiveCharacterTextSplitter(...): Splits the loaded PDF into manageable chunks before processing.
retrieval_chain = create_retrieval_chain(retriever, document_chain): Creates a pipeline where the retrieved chunks are passed to the language model along with the user’s question.

9. Answer Generation (Language Model)

    
Description:

The retrieved context is passed to a large language model (LLM) that generates a response based on the given context.


Components:


LLM (Language Model): 

In this case, ChatGroq, a large language model, processes the user’s question and the retrieved context to produce a coherent answer.
Prompt Template: The prompt template defines how the retrieved context and the question are framed for the model. 


Flow:


The context and user query are passed to the language model, and the model generates an answer based on this combined information.


11. Response Display

    
Description:

The generated answer from the model is displayed in the Streamlit interface, along with the user’s original query.


Components:


st.markdown(f"<div class='chat-message assistant'>{response['answer']}</div>", unsafe_allow_html=True): Displays the chatbot’s response in a chat-like format.
The user’s input and the assistant’s response are stored in session history, allowing the conversation to flow naturally.


13. Chat History

    
Description: The chat history, including all previous prompts and answers, is stored and displayed dynamically.
Components:
Session State: This stores the chat history:



Flow:


Every prompt and response is saved and displayed as part of the chat history.


15. Document Similarity Search (Expander)

    
Description:

Along with the chatbot’s answer, the relevant document chunks (retrieved context) are displayed in an expander. This shows how the answer was derived from the context.


Components:


st.expander("Document Similarity Search"):: This creates an expandable section to show the raw text from the document chunks.
Document content from response["context"]: Displays the text from the retrieved chunks.


Summary of the Flow:


User Prompt → 2. Prompt Processing → 3. Document Retrieval → 4. Context Generation → 5. Answer Generation by LLM → 6. Response Display → 7. Chat History Storage → 8. Document Similarity Search Display
This entire flow allows the chatbot to deliver contextual, accurate, and fact-based answers using the RAG approach.
