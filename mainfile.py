import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from PyPDF2 import PdfReader
import torch

st.set_page_config(page_title="RAG PDF Q&A", layout="wide")
st.title("RAG Model - PDF Question Answering")

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'llm' not in st.session_state:
    st.session_state.llm = None


@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.3,
        top_p=0.95
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


with st.sidebar:
    st.header("Configuration")

    if st.button("Load Model", use_container_width=True):
        with st.spinner("Loading LLM (first time takes 1-2 minutes)..."):
            st.session_state.llm = load_llm()
            st.success("Model loaded successfully!")

    st.divider()

    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

    if st.button("Process PDF", use_container_width=True) and uploaded_file:
        if not st.session_state.llm:
            st.error("Please load the model first!")
        else:
            with st.spinner("Processing PDF..."):
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                vectorstore = FAISS.from_texts(chunks, embeddings)
                st.session_state.vectorstore = vectorstore

                qa_chain = RetrievalQA.from_chain_type(
                    llm=st.session_state.llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )

                st.session_state.qa_chain = qa_chain
                st.success("PDF processed successfully!")

st.header("Ask Questions")

if st.session_state.qa_chain:
    question = st.text_input("Enter your question:", placeholder="What is this document about?")

    if st.button("Get Answer", use_container_width=True) and question:
        with st.spinner("Generating answer..."):
            result = st.session_state.qa_chain({"query": question})

            st.subheader("Answer:")
            st.info(result['result'])

            st.subheader("Sources:")
            for i, doc in enumerate(result['source_documents']):
                with st.expander(f"Source {i + 1}"):
                    st.text(doc.page_content)
else:
    st.warning("Please load the model and process a PDF first using the sidebar.")