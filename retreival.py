from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedder=HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
presist="db/chromadb"

def retriever():
    chdb=Chroma(
        persist_directory=presist,
        embedding_function=embedder,
        collection_metadata={"hnsw:space":"cosine"}
    )
    retriever=chdb.as_retriever(

        search_kwargs={"k":3}
    )
    relevant_doc=retriever.invoke("tell me about pakistan economy")
    return relevant_doc

print(retriever())