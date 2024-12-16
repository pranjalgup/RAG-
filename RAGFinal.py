import os
import string
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma


# Load environment variables
load_dotenv()

# Fetch OpenAI API key from .env
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please check your .env file.")

# Constants
DATA_PATH = r"D:\common code\unstructured RAG\demo_Data"
CHROMA_PATH = "chromatest"


# Document Loader
class DocumentLoader:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def load_documents(self):
        all_text = ""
        try:
            # List all PDF files in the directory
            pdf_files = [f for f in os.listdir(self.directory_path) if f.endswith('.pdf')]

            if not pdf_files:
                print("No PDF files found in the directory.")
                return None

            for pdf_file in pdf_files:
                file_path = os.path.join(self.directory_path, pdf_file)
                print(f"Reading file: {file_path}")
                reader = PdfReader(file_path)
                # Extract text from all pages in the PDF
                all_text += "".join(page.extract_text() for page in reader.pages if page.extract_text())

            return all_text.strip()
        except Exception as e:
            print(f"Error loading documents: {e}")
            return None


# Preprocessing: Remove punctuation and extra spaces
class Preprocessing:
    def __init__(self, document):
        self.document = document

    def remove_punctuation(self):
        return self.document.translate(str.maketrans('', '', string.punctuation))

    def remove_extra_spaces(self, text):
        return " ".join(text.split())


# Text Chunking: Chunking text using various splitters
class TextChunker:
    def __init__(self, embeddings, db_path="CHROMA_PATH", chunk_size=1000, chunk_overlap=200):
        self.embeddings = embeddings
        self.db_path = db_path
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.db_path
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_methods = {
            "recursive": RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap),
            "markdown": MarkdownTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap),
            "semantic": SemanticChunker(embeddings, breakpoint_threshold_type="gradient")
        }

    def collection_exists(self, collection_name):
        try:
            self.vectorstore._client.get_collection(collection_name)
            print(f"collection-name-----------------------------",collection_name)
            return True
        except Exception as e:
            print(f"Error checking collection: {e}")
            return False

    def process_pdf(self, document_text, chunking_method):

        if chunking_method not in self.chunking_methods:
            document_loader = DocumentLoader(DATA_PATH)
            document_text = document_loader.load_documents()
            raise ValueError(f"Unknown chunking method: {chunking_method}")

        text_splitter = self.chunking_methods[chunking_method]
        chunks = text_splitter.split_text(document_text)
        return chunks

    def store_in_chroma(self, chunks, collection_name):
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path,
        )
        vectorstore.add_texts(chunks)
        vectorstore.persist()
        print(f"Stored chunks in collection: {collection_name}")

    def get_retriever(self, collection_name):
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path,
        )
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        return retriever


# RAG System that combines retrieval and generation
class RAGSystem:
    def __init__(self, embeddings, chroma_handler):
        self.embeddings = embeddings
        self.chroma_handler = chroma_handler

    def perform_rag(self, query, collection_name):
        retriever = self.chroma_handler.get_retriever(collection_name)
        if retriever:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=False,
            )
            # Perform RAG and extract only the 'result' from the output
            response = qa_chain({"query": query})
            return response.get('result', 'No result found')
        else:
            print(f"Cannot perform RAG as the collection {collection_name} is not found.")
            return None

def main():
    # Initialize embeddings and Chroma handler
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    chroma_handler = TextChunker(embeddings=embeddings, db_path=CHROMA_PATH)

    # Load documents
    document_loader = DocumentLoader(DATA_PATH)
    document_text = document_loader.load_documents()
    if not document_text:
        print("No documents loaded. Exiting.")
        return

    # Process the document text using different chunking methods
    for method in chroma_handler.chunking_methods:
        collection_name = f"{method}_chunks"
        if not chroma_handler.collection_exists(collection_name):
            print(f"Collection '{collection_name}' does not exist. Creating it...")
            chunks = chroma_handler.process_pdf(document_text, method)
            chroma_handler.store_in_chroma(chunks, collection_name)
        else:
            print(f"Collection '{collection_name}' already exists. Skipping chunking.")

    query = input("Write your query?\n")
    rag_system = RAGSystem(embeddings=embeddings, chroma_handler=chroma_handler)
    response = rag_system.perform_rag(query, "markdown_chunks")
    #here you can select which chunks to use. 
    print(response)


if __name__ == "__main__":
    main()
