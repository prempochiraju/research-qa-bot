import os
import sys
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Research Q&A Bot", page_icon="🤖", layout="centered")
st.title("🤖 Research Paper Q&A Bot")
st.caption("Built by Prem Pochiraju · LangChain + ChromaDB + OpenAI RAG Pipeline")

with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    uploaded_files = st.file_uploader("Upload PDF papers", type=["pdf"], accept_multiple_files=True)
    index_btn = st.button("📥 Index Papers", use_container_width=True)
    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("- LangChain · ChromaDB\n- OpenAI Embeddings\n- GPT-3.5-turbo\n- Streamlit")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def build_index(files, api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp())
    all_docs = []
    for f in files:
        p = tmp / f.name
        p.write_bytes(f.read())
        all_docs.extend(PyPDFLoader(str(p)).load())
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(all_docs)
    vs = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(model="text-embedding-3-small")
    )
    shutil.rmtree(tmp)
    return vs

def ask(vs, question, api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    docs = vs.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_template(
        "You are a research assistant. Answer using only the context below.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    )
    chain = prompt | ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    seen = set()
    sources = []
    for d in docs:
        src = Path(d.metadata.get("source", "?")).name
        page = d.metadata.get("page", "?")
        key = f"{src}:{page}"
        if key not in seen:
            sources.append(f"**{src}** (page {page})")
            seen.add(key)
    return answer, sources

if index_btn:
    if not api_key:
        st.sidebar.error("Please enter your OpenAI API key.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    else:
        with st.spinner("Indexing papers..."):
            st.session_state.vectorstore = build_index(uploaded_files, api_key)
            st.session_state.messages = []
        st.sidebar.success(f"✅ Indexed {len(uploaded_files)} paper(s)!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your papers..."):
    if st.session_state.vectorstore is None:
        st.warning("Please upload papers and click 'Index Papers' first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching papers..."):
                answer, sources = ask(st.session_state.vectorstore, prompt, api_key)
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"• {s}")
        st.session_state.messages.append({"role": "assistant", "content": answer})