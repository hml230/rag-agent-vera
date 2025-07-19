"""Vector storage class for papers embeddings"""
import os
import getpass
from pathlib import Path

import yaml
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from storage import PapersDB

load_dotenv()
if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")


config_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
PERSIST_PATH = config['db_paths']['chroma']


class PaperVectorStore:
    """Vector storage to embed and store/persist embeddings"""
    def __init__(self, papers_db: PapersDB):
        self.papers_db = papers_db
        self.embeddings_model = MistralAIEmbeddings(model="mistral-embed")
        self.vstore = Chroma(collection_name="research_papers_collection",
                            embedding_function=self.embeddings_model,
                            persist_directory=PERSIST_PATH,
                            )

    def build_storage(self, sql_query_limit = 500):
        """Load papers from DB and embed/add to storage"""
        cursor = self.papers_db.conn.execute(
            """
            SELECT id, title, abstract 
            FROM papers ORDER BY created_at DESC LIMIT ?
            """, (sql_query_limit,)
            )
        docs = []
        for pid, title, abstract in cursor.fetchall():
            content = f"{title}\n\n{abstract or ''}"
            docs.append(Document(page_content=content, metadata={"id": pid, "title": title}))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(doc for doc in docs)
        _ = self.vstore.add_documents(documents=all_splits)
