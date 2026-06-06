import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv

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

    


def main():
    print("this is main")
    doc_loader("./documents/")




if __name__ == "__main__":
    main()

    