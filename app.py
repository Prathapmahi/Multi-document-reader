import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Do not provide incorrect information.

    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("\n🗨️ Reply:", response["output_text"])

def main():
    st.set_page_config("📄 AI PDF Assistant", page_icon=":robot_face:")
    
    st.markdown(
        """
        <style>
            @keyframes glow {
                0% { text-shadow: 0 0 5px #ffffff; }
                50% { text-shadow: 0 0 20px #00ffcc; }
                100% { text-shadow: 0 0 5px #ffffff; }
            }
            .animated-title {
                font-size: 30px;
                font-weight: bold;
                text-align: center;
                color: #00ffcc;
                animation: glow 1.5s infinite alternate;
            }
        </style>
        <div class='animated-title'>📄 AI-Powered PDF Assistant 🤖</div>
        """,
        unsafe_allow_html=True
    )

    user_question = st.text_input("Ask a question from the uploaded PDFs 📜📝")
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.image("img/Robot.jpg")
        st.write("---")
        st.title("📂 Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload PDF files & click 'Process'", accept_multiple_files=True)
        if st.button("📥 Process PDFs"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Processing Complete!")
        
        st.write("---")
        st.image("img/ptp.jpg")
        st.write("Created by @ Prathap")
    
    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/Prathap-Elumalai/Prathap" target="_blank" style="color: #00ffcc; text-decoration: none;">Prathap</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    
