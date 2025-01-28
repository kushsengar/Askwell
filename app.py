import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import time
import pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from htmlTemplates import css, bot_template, user_template
from together import Together
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Initialize Together.ai client
client = Together(api_key=os.getenv("TOGETHER_AI_API_KEY"))

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to set up Pinecone
def Pinecone_setup():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

    # Initialize the Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index_name = "chatwithdocument"  # Change if desired
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)
    return index_name, index

# Function to retrieve relevant documents for a query
def retrieve_query(query, embeddings, index, k=3):
    query_embedding = embeddings.embed_query(query)
    matching_results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    return [(result['metadata']['text'], result['score']) for result in matching_results['matches']]


def prepare_prompt(query, context):
    context_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(context)])
    prompt = f"""You are an assistant with access to the following context:
{context_text}

Now, answer the following query:
Query: {query}
Answer:"""
    return prompt


def get_response_from_llm(prompt, client):
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None
        
# Function to query the document
def query_document(query, embeddings, index, k=2):
    context = retrieve_query(query, embeddings, index, k)
    prompt = prepare_prompt(query, context)
    response = get_response_from_llm(prompt, client)
    return response

# Function to handle user input
def handle_userinput(user_question):
    if "index" in st.session_state and "embeddings" in st.session_state:
        index = st.session_state.index
        embeddings = st.session_state.embeddings
        response = query_document(user_question, embeddings, index)
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        for i, message in enumerate(st.session_state.chat_history):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
    else:
        st.error("Please process your documents first.")

# Main function
def main():
    st.set_page_config(page_title="Askwell", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with your document using Askwell :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)

                index_name, index = Pinecone_setup()
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                st.session_state.index = index  # Store index in session state
                st.session_state.embeddings = embeddings  # Store embeddings in session state

                vectorstore = LangchainPinecone.from_texts(
                    texts=text_chunks, embedding=embeddings, index_name=index_name
                )
                st.session_state.pinecone_index = vectorstore
                st.session_state.conversation = True
                st.success("Documents processed successfully!")

if __name__ == '__main__':
    main()
