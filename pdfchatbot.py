import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, respond with "Answer is not available in the context".
    
    Context:
    {context}?
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.markdown(f"""
    <div style="background-color:#e3f2fd;padding:10px;border-radius:10px;margin-top:10px;color:#0d47a1;font-family:Arial;font-size:16px;">
        <b>Reply:</b> {response["output_text"]}
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="📄")
    st.markdown("""
        <style>
            body {
                background-color: #f5f5f5;
                font-family: 'Arial', sans-serif;
            }
            .stTextInput > div > div > input {
                font-size: 16px;
                font-family: 'Arial', sans-serif;
            }
        </style>
        <h1 style='text-align: center; color: #1565c0; font-family: Arial;'>📄 Chat with PDF using Gemini 💬</h1>
        <hr style='border: 1px solid #ddd;'>
    """, unsafe_allow_html=True)
    
    user_question = st.text_input("💡 Ask a Question from the PDF Files:", key="user_input")
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.markdown("""
            <h2 style='text-align: center; color: #1e88e5; font-family: Arial;'>📂 Menu</h2>
        """, unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("🚀 Submit & Process"):
            with st.spinner("⏳ Processing PDFs... Please wait"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Processing Complete!")

if __name__ == "__main__":
    main()
