import os
import sys
import logging
import asyncio
import threading
import queue
import pickle
from typing import Any, Dict, Optional, Union, Mapping
import uuid
from uuid import UUID
import faiss
import hashlib

import openai
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter  # generic
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatGeneration,
    ChatResult,
)
from langchain.chains import (
    LLMChain,
    ConversationalRetrievalChain,
)
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import (
    BaseCallbackHandler,
    AsyncCallbackHandler,
)
from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
    AsyncCallbackManagerForChainRun,
)

logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1280, chunk_overlap=200)
embedding_model = {
    'name': 'openai',
    'func': None,
}

openai_env = {
    'api_key': None,
    'api_base': None,
}

openai_model = {
    'name': 'gpt-3.5-turbo',
    'max_tokens': 4096,
    'max_prompt_tokens': 3096,
    'max_response_tokens': 1000
}

_queue = queue.Queue()

def setup_openai_env(api_base=None, api_key=None):
    if not openai_env['api_base']:
        openai_env['api_base'] = api_base
    if not openai_env['api_key']:
        openai_env['api_key'] = api_key
    openai.api_base = openai_env['api_base']
    openai.api_key = openai_env['api_key']
    openai.api_version = None
    return (openai_env['api_base'], openai_env['api_key'])


def setup_openai_model(model):
    logger.debug(model)
    openai_model.update(model)
    logger.debug(model)


# class OutputStreamingCallbackHandler(AsyncCallbackHandler):
class OutputStreamingCallbackHandler(BaseCallbackHandler):
    send_token: bool = False

    # make it a producer to send us reply
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self.send_token:
            _queue.put(token)
            # sys.stdout.write(token)
            # sys.stdout.flush()

    def on_chain_start(self, serialized, inputs, **kwargs) -> Any:
        """run when chain start running"""
        # don't stream the output from intermedia steps
        logger.debug('****** launch chain %s', serialized)
        if serialized['name'] == 'StuffDocumentsChain':
            logger.debug('start output streamming')
            self.send_token = True

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any,) -> None:
        """Run when chain ends running."""
        # _queue.put(-1)
        # return await super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_llm_error( self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when LLM errors."""
        _queue.put(-1)

    def on_chain_error( self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when chain errors."""
        _queue.put(-1)


OSC = OutputStreamingCallbackHandler()


class EmbeddingModel:
    def __init__(self):
        self.name = None
        self._function = None

    @property
    def function(self):
        """embedding function of the model"""
        if not self._function:
            setup_openai_env()
            self.name = 'openai'
            self._function = OpenAIEmbeddings()
        return self._function


class ChatModel:
    def __init__(self):
        self.name = None
        self._model = None

    @property
    def model(self):
        if not self._model:
            api_base, api_key = setup_openai_env()
            self.name = 'open_ai'
            max_response_tokens = openai_model['max_prompt_tokens']
            if max_response_tokens > 1024:
                max_response_tokens = 1024
            self._model = ChatOpenAI(
                api_key=api_key,
                api_base=api_base,
                model_name=openai_model['name'],
                max_tokens=max_response_tokens,
                streaming=True,
                # callbacks=[OSC],
            )
        return self._model


embedding_model = EmbeddingModel()
chat_model = ChatModel()

def pickle_faiss(db):
    idx = faiss.serialize_index(db.index)
    pickled = pickle.dumps((db.docstore, db.index_to_docstore_id, idx))
    return pickled

def unpick_faiss(pickled, embedding_func = None):
    if not embedding_func:
        embedding_func = embedding_model.function
    docstore, index_to_docstore_id, idx = pickle.loads(pickled)
    index = faiss.deserialize_index(idx)
    db = FAISS(embedding_func.embed_query, index, docstore, index_to_docstore_id)
    return db

def get_embedding_document(file, mime):
    """return a pickled faiss vectorsotre"""

    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types
    loaders = {
        'text/plain': TextLoader,
        'application/pdf': PyPDFLoader,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': Docx2txtLoader,
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': UnstructuredPowerPointLoader,
    }

    loader = loaders[mime](file)
    docs = loader.load()

    embeddings_function = embedding_model.function

    for doc in docs:
        hash_str = str(hashlib.md5(str(doc).encode()).hexdigest())
        doc.metadata['hash'] = hash_str  # track where chunk from
    documents = text_splitter.split_documents(docs)
    db = FAISS.from_documents(documents, embeddings_function)

    return pickle_faiss(db)


condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question (in the same language of Follow Up Input):"""
MY_CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)


def langchain_doc_chat(messages):
    """use langchain to process a list of messages"""

    db = messages['faiss_store']
    retriever = db.as_retriever(
        search_type="mmr",
        #search_type="similarity",
        search_kwargs={
            'k': 3,
        },
    )

    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True, k=32)
    for msg in messages['messages']:
        if msg.get('role', '') == 'assistant':
            memory.chat_memory.add_ai_message(msg['content'])
        else:  # user or system message
            memory.chat_memory.add_user_message(msg['content'])

    question_generator = LLMChain(
        llm=chat_model.model,
        prompt=MY_CONDENSE_QUESTION_PROMPT
    )
    doc_chain = load_qa_chain(
        llm=chat_model.model,
        chain_type="map_reduce",
    )
    chain = ConversationalRetrievalChain(
        retriever=retriever,
        memory=memory,
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
    )

    results = []
    msgs = messages['messages']
    q = msgs[-1]['content']
    logger.debug(q)
    OSC.send_token = False

    async def do_chain():
        result = await chain.acall(
            {'question': q},
            callbacks=[OSC],
        )
        _queue.put(-1)
        return result

    def ctx_mgr():
        result = asyncio.run(do_chain())
        results.append(result)

    thread = threading.Thread(target=ctx_mgr)
    thread.start()

    while True:
        item = _queue.get()
        # logger.debug('>>>>\n>>>> partial item %s', item)
        if item == -1:
            logger.debug('langchan done')
            yield {
                'content': item,
                'status': 'done',
            }
            _queue.task_done()
            break
        yield {
            'content': item,
            'status': None,
        }
        _queue.task_done()

    thread.join()
    logger.debug('langchan exit with %s', results[0])  # should output a coroutine

    return
