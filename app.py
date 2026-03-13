"""
Research Paper Q&A Bot

This script creates a simple question–answering system for research papers.
It loads PDF files from a local folder, splits the text into smaller chunks,
and generates embeddings using OpenAI's embedding model. These embeddings
are stored in a Chroma vector database for efficient similarity search.

When a user asks a question, the system retrieves the most relevant document
sections and sends them to a language model to generate an answer based only
on the retrieved context. The program also prints the source file names and
page numbers used to produce the response.

Requirements:
- OpenAI API key set as environment variable (OPENAI_API_KEY)
- PDF papers placed inside the ./papers directory

Main Components:
1. Indexing PDFs and creating embeddings
2. Loading the vector database
3. Retrieving relevant document chunks
4. Generating answers using a language model
@author Prem Pochiraju
"""


import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PAPERS_DIR = "./papers"
CHROMA_DIR = "./chroma_db"

def index_papers():
    print("\nLoading PDFs from ./papers ...")
    docs = DirectoryLoader(
        PAPERS_DIR, glob="**/*.pdf",
        loader_cls=PyPDFLoader, show_progress=True
    ).load()
    if not docs:
        print("No PDFs found. Add PDFs to ./papers first.")
        sys.exit(1)
    print(f"{len(docs)} pages loaded.")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(docs)
    print(f"{len(chunks)} chunks. Embedding now (takes ~30 sec)...")
    vs = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=CHROMA_DIR
    )
    print("Done!")
    return vs

def load_index():
    print("Loading existing index...")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )

def ask(vs, question):
    docs = vs.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_template(
        "You are a research assistant. Answer using only the context below.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
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
            sources.append(f"    {src} (page {page})")
            seen.add(key)
    return answer, sources

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print('Set your API key first:')
        print('   $env:OPENAI_API_KEY="sk-..."')
        sys.exit(1)

    if Path(CHROMA_DIR).exists():
        choice = input("Found existing index. Re-index papers? (y/N): ").strip().lower()
        if choice == "y":
            import shutil
            shutil.rmtree(CHROMA_DIR)
            vs = index_papers()
        else:
            vs = load_index()
    else:
        vs = index_papers()

    print("\n" + "="*55)
    print("  Research Paper Q&A Bot  |  type 'exit' to quit")
    print("="*55 + "\n")

    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break

        answer, sources = ask(vs, q)
        print(f"\n{answer}\n")
        if sources:
            print("Sources:")
            for s in sources:
                print(s)
        print()

if __name__ == "__main__":
    main()
