import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Step 1: Load and process PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Step 2: Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Step 3: Create vector store
def create_vector_store(chunks):
    # Use OpenAI embeddings instead of HuggingFace embeddings
    from langchain_community.embeddings import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base="https://api.deepseek.com",
        model="embedding-2"
    )
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

# Step 4: Set up DeepSeek API
def setup_deepseek():
    llm = OpenAI(
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base="https://api.deepseek.com",
        model_name="deepseek-chat",
        temperature=0.7,
    )
    return llm

# Step 5: Create RAG pipeline
def create_rag_pipeline(vector_store, llm):
    prompt_template = """Answer the question based on the context. Be brief and accurate.

    Context: {context}

    Question: {question}

    Answer: """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
    )
    return qa_chain

# Function to process PDF and answer a query
def answer_query(pdf_path, query):
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    llm = setup_deepseek()
    qa_chain = create_rag_pipeline(vector_store, llm)
    response = qa_chain.run(query)
    return response
