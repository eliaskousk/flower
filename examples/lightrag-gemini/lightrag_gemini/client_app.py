"""LightRAG Gemini: Flower ClientApp."""

import asyncio
from typing import Optional

from flwr.client import ClientApp
from flwr.common import ConfigRecord, Context, Message, RecordDict

from .lightrag_client import LightRAGClient

# Global client instance and event loop
_client_instance: Optional[LightRAGClient] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get the existing event loop or create a new one."""
    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_event_loop)
    return _event_loop


def run_async(coro):
    """Run an async coroutine using the shared event loop."""
    loop = get_or_create_event_loop()
    return loop.run_until_complete(coro)


# Flower ClientApp
app = ClientApp()


@app.query("init")
def init(msg: Message, context: Context):
    """Initialize LightRAG on the client."""
    global _client_instance

    config = msg.content["config"]
    _client_instance = LightRAGClient(str(context.node_id), config)

    # Run async initialization using shared loop
    run_async(_client_instance.initialize(task_type="RETRIEVAL_DOCUMENT", first=True))

    return Message(
        RecordDict({"status": ConfigRecord({"initialized": True, "node_id": context.node_id})}),
        reply_to=msg
    )


@app.query("insert")
def insert(msg: Message, context: Context):
    """Insert documents into LightRAG."""
    global _client_instance

    if not _client_instance:
        return Message(
            RecordDict({"error": ConfigRecord({"message": "LightRAG not initialized"})}),
            reply_to=msg
        )

    documents = msg.content["config"]["documents"]

    # Run async document insertion using shared loop
    num_inserted = run_async(_client_instance.insert_documents(documents))

    return Message(
        RecordDict({
            "status": ConfigRecord({
                "inserted": num_inserted,
                "node_id": context.node_id
            })
        }),
        reply_to=msg
    )


@app.query("query")
def query(msg: Message, context: Context):
    """Query the LightRAG instance."""
    global _client_instance

    if not _client_instance:
        return Message(
            RecordDict({"error": ConfigRecord({"message": "LightRAG not initialized"})}),
            reply_to=msg
        )

    config: ConfigRecord = msg.content["config"]
    question = config["question"]
    mode = config.get("mode", "hybrid")
    top_k = config.get("top_k", 5)

    # Re-initialize for querying (without RETRIEVAL_DOCUMENT task type)
    run_async(_client_instance.initialize(task_type="RETRIEVAL_QUESTION", first=False))

    # Run async query using shared loop
    result = run_async(_client_instance.query(question, mode, top_k))

    return Message(
        RecordDict({
            "result": ConfigRecord(result)
        }),
        reply_to=msg
    )
