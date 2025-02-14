# Chat with Your Documents Using Askwell
Askwell is a Retrieval-Augmented Generation (RAG) pipeline application designed to interact with documents using natural language queries. At its core, Askwell leverages transformer models, advanced embedding techniques, and vector stores to retrieve and summarize document information effectively.

# About Transformer Models
Transformer models are at the heart of modern natural language processing (NLP). They excel in understanding context through self-attention mechanisms, enabling them to process and generate human-like text. Askwell uses the transformer model's capabilities to generate responses by combining document context with user queries.

# Libraries and Components
PyPDF2: Used to extract raw text from uploaded PDF files. It efficiently parses and retrieves document content.

CharacterTextSplitter: This utility chunks the extracted text into manageable pieces, ensuring an optimal input size for embedding and querying while preserving context.

Google GenerativeAI Embeddings: Converts text into high-dimensional vectors, capturing semantic meaning. These embeddings enable accurate retrieval of document-relevant sections.

Pinecone: Acts as the vector store to index and query embeddings. It provides fast and scalable similarity searches, making it the backbone of the RAG pipeline.

Streamlit: Powers the user interface for document upload, processing, and interactive chat with a streamlined experience.

Together.ai: Facilitates interaction with the LLM (large language model) for generating responses based on the retrieved document context.

# How the RAG Pipeline Works
![PDF-LangChain](https://github.com/user-attachments/assets/72dbd425-6025-4025-9254-7ead3e4e73da)
The application follows these steps to provide responses to your questions:

 1.PDF Loading: The app reads multiple PDF documents and extracts their text content.

 2.Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

 3.Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

 4.Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

 5.Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.

# Users ask questions via the chat interface.
The RAG pipeline retrieves relevant text chunks from Pinecone based on the query embedding.
A prompt combining the query and retrieved context is sent to the LLM for a response.
Response Generation:

The LLM processes the prompt and generates an answer.
The result is displayed in the chat interface for the user.
Features
Interactive Chat: Users can interact with their documents in natural language.
Multi-Document Support: Upload and process multiple PDFs simultaneously.
Efficient Retrieval: Pinecone ensures quick retrieval of relevant content.
Scalable Architecture: Designed to handle large documents and queries seamlessly.
Usage
Clone the repository and install dependencies using requirements.txt.
Run the application with streamlit run app.py.
Upload your PDFs and start querying your documents.
With Askwell, make your documents smarter and accessible in ways never before possible!

