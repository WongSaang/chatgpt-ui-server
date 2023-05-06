"""
Message tools
"""
import os
import sys
import pickle
import json
import logging
import hashlib
from typing import Any, Dict, List, Optional

import arxiv
from langchain.schema import Document
from .models import Conversation, Message, Setting, Prompt, EmbeddingDocument
from .llm import text_splitter, embedding_model, pickle_faiss

logger = logging.getLogger(__name__)

from utils.search_prompt import compile_prompt
from utils.duckduckgo_search import web_search, SearchRequest


def _web_search(message, args):
    '''search the web
    '''
    search_results = web_search(SearchRequest(message, ua=args['ua']), num_results=5)
    message_content = compile_prompt(search_results, message, default_prompt=args['default_prompt'])
    return message_content


arxiv_client = arxiv.Client(
  page_size = 100,
  delay_seconds = 3,
  num_retries = 2,
)

def _hacky_hash(some_string):
    _hash = hashlib.md5(some_string.encode("utf-8")).hexdigest()
    return _hash

def _arxiv_load(
    query: Optional[str] ='',
    id_list: Optional[str|List[str]] = [],
    max_results: int = 2,
    sort_by: Optional[Any] = arxiv.SortCriterion.Relevance,
    papers_dir: Optional[str] = ".papers",
    load_all_available_meta: bool = False,
) -> List[Document]:
    """
    Run Arxiv search and get the PDF documents plus the meta information.
    See https://lukasschwab.me/arxiv.py/index.html#Search

    Returns: a list of documents with the document.page_content in PDF format

    """

    if isinstance(id_list, str):
        id_list = id_list.split(',')

    if query:
        query = query[:2048]

    try:
        import fitz
    except ImportError:
        raise ValueError(
            "PyMuPDF package not found, please install it with "
            "`pip install pymupdf`"
        )

    try:
        docs: List[Document] = []
        arxiv_search = arxiv.Search(
            query=query,
            id_list=id_list,
            max_results=max_results,
            sort_by=sort_by
        )
        search_results = list(arxiv_client.results(arxiv_search))

        if not os.path.exists(papers_dir):
            os.makedirs(papers_dir)

        for result in search_results:
            try:
                paper = result
                filename = f"{_hacky_hash(result.title)}.pdf"
                doc_file_name: str = os.path.join(papers_dir, filename)
                paper.download_pdf(dirpath=papers_dir, filename=filename)
                logging.debug(f"> Downloading {filename}...")
                with fitz.open(doc_file_name) as doc_file:
                    text: str = "".join(page.get_text() for page in doc_file)
                    add_meta = (
                        {
                            "entry_id": result.entry_id,
                            "published_first_time": str(result.published.date()),
                            "comment": result.comment,
                            "journal_ref": result.journal_ref,
                            "doi": result.doi,
                            "primary_category": result.primary_category,
                            "categories": result.categories,
                            "links": [link.href for link in result.links],
                        }
                        if load_all_available_meta
                        else {}
                    )
                    doc = Document(
                        page_content=text,
                        metadata=(
                            {
                                "Published": str(result.updated.date()),
                                "Title": result.title,
                                "Authors": ", ".join(
                                    a.name for a in result.authors
                                ),
                                "Summary": result.summary,
                                **add_meta,
                            }
                        ),
                    )
                    docs.append(doc)
            except FileNotFoundError as f_ex:
                logger.debug(f_ex)

        # Delete downloaded papers
        try:
            for f in os.listdir(papers_dir):
                os.remove(os.path.join(papers_dir, f))
                logging.debug(f"> Deleted file: {f}")
            #os.rmdir(papers_dir)
            logging.debug(f"> Deleted directory: {papers_dir}")
        except OSError:
            print("Unable to delete files")

        return docs

    except Exception as ex:
        logger.debug("Error on arxiv: %s", ex)
        return []


def _arxiv(message, args):
    """Dowload arxiv PDF and embedding it"""
    
    from langchain.vectorstores import FAISS

    ID =  message.strip()
    message = '[arxiv] ' + ID

    logger.debug('process arxiv message %s : %s', message, args)

    try:
        docs = _arxiv_load(id_list=[ID], max_results=1)
        if len(docs) == 0:
            raise RuntimeError()
    except Exception as e:
        logger.error('cannot download %s', ID)
        return f'Failed to download ArXiv: {ID}'

    logger.debug('Download %d arxiv documents', len(docs))
    documents = text_splitter.split_documents(docs)
    db = FAISS.from_documents(documents, embedding_model.function)
    faiss_store = pickle_faiss(db)

    doc_obj = EmbeddingDocument(
        user=args['user'],
        faiss_store=faiss_store,
        title=docs[0].metadata['Title'],
    )
    doc_obj.save()
    args['embedding_doc_id'] = doc_obj.id
    args['doc_title'] = docs[0].metadata['Title']

    message += ' [Title] ' + docs[0].metadata['Title']

    return message


TOOL_LIST = {
    'web_search': _web_search,
    'arxiv': _arxiv,
}