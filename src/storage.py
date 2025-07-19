"""Class definition for a SQLlite database"""
import sqlite3
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


config_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
SQL_PATH = config['db_paths']['sqlite']


class PapersDB:
    """SQLite-based storage for queried papers""" 
    def __init__(self, db_path: str = SQL_PATH + "papers.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.setup_tables()

    def setup_tables(self):
        """Create database tables if they don't exist"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                abstract TEXT,
                content TEXT,
                source TEXT,
                url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def store_paper(self, paper_data: Dict[str, Any]) -> int:
        """Create new paper records in DB with corresponding schema"""
        cursor = self.conn.execute('''
            INSERT INTO papers (title, abstract, content, source, url)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            paper_data.get('title', ''),
            paper_data.get('summary', ''),
            paper_data.get('content', ''),
            paper_data.get('source', ''),
            paper_data.get('url', '')
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_paper(self, paper_id: int) -> Optional[tuple]:
        """Retrieve paper by ID"""
        cursor = self.conn.execute(
            '''SELECT id, title, abstract, content, source, url 
            FROM papers WHERE id = ?''',
            (paper_id,)
        )
        return cursor.fetchone()

    def get_paper_count(self) -> int:
        """Get total number of papers"""
        cursor = self.conn.execute('SELECT COUNT(*) FROM papers')
        return cursor.fetchone()[0]
