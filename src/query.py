"""Class definiton for API call functions"""
from typing import List
import logging
import time

import arxiv

from storage import PapersDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FreeDataCollector:
    """Collect papers from free APIs"""
    def __init__(self, storage: PapersDB):
        self.storage = storage

    def fetch_papers(self, query: str, max_results=200):
        """Call API to query papers via the arxiv object"""
        try:
            logger.info("Querying %s papers from arxiv", max_results)
            search = arxiv.Search(query=query, max_results=max_results)
            logger.info("API successfully returned results for query %s", query)
            return [{
                "title": result.title,
                "summary": result.summary,
                "url": result.pdf_url,
                "published": result.published,
            } for result in search.results()]

        except Exception as e:
            logger.error("Error %s fetching paper from API", e)
            return []

    def query_data(self, topics: List[str] = None) -> int: # type: ignore
        """Build dataset from retrieved papers"""
        if topics is None:
            topics = ['machine learning', 'biology', 'chemistry']

        all_papers = []

        # Collect from arXiv
        for topic in topics:
            papers = self.fetch_papers(topic)
            all_papers.extend(papers)
            time.sleep(4)  # API call gaps as per usage policy

        stored_count = 0
        for paper in all_papers:
            try:
                # Store in database
                self.storage.store_paper(paper)
                stored_count += 1
                if stored_count % 10 == 0:
                    logger.info("Stored %s papers...", stored_count)

            except Exception as e:
                logger.error("Error storing paper: %s", e)
                continue

        logger.info("Successfully stored %s papers", stored_count)
        return stored_count
