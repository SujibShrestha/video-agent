
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def get_embeddings():
    return HuggingFaceEndpointEmbeddings(model=EMBEDDING_MODEL)

def bulid_vector_store(transcript:str)->Chroma:
    print("Building vector store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    chunks = splitter.split_text(transcript)
    docs=[
        Document(page_content= chunk,metadata={'chunk-index':i})
                 for i ,chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents= docs,
        embedding= embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

    return vector_store


def load_vector_store():
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name = COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def get_retriever(vector_store:Chroma,k:int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k':k}
    )

