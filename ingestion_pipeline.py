import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def doc_loader(docpath):
    loader=DirectoryLoader(
        path=docpath,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding":"utf-8"}
    )
    document=loader.load()
    for doc in document:
        print(doc.metadata)
        print(f"document length = {len(doc.page_content)}")
    return document

def chunker(documents,chunksize, chunkoverlap):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunksize,
        chunk_overlap=chunkoverlap
    )
    chunks=splitter.split_documents(documents)
    print(len(chunks))
    return chunks    

def vector_store(chunks, presist_loaction):
    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)   
    store=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=presist_loaction,
        collection_metadata={"hnsw:space":"cosine"}
        
    )
   
    return store


def main():
    print("this is main")
    documents=doc_loader("./documents/")
    chunks =chunker(documents, 1000, 200)
    store= vector_store(chunks, "db/chromadb")






if __name__ == "__main__":
    main()

    