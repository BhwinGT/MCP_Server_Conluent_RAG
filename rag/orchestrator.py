"""
orchestrator.py
---------------
Main Q&A loop:
  1. Takes a user question
  2. Calls the MCP server (qdrant_server.py) to retrieve relevant doc chunks
  3. Builds a prompt with retrieved context
  4. Streams the answer from Qwen3 via Ollama

Requires qdrant_server.py to already be running in another terminal.

Run with:
    python rag/orchestrator.py
"""

import os
import asyncio

from ollama import Client
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from dotenv import load_dotenv

load_dotenv()

INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "gemma4:31b")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
TOP_K           = int(os.getenv("TOP_K_RESULTS", "5"))
MCP_HOST        = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT        = os.getenv("MCP_PORT", "8000")

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
_local_client = Client(host=OLLAMA_HOST)

# Inference (chat) goes to Ollama Cloud if an API key is set; embedding stays on the local/container Ollama
_USE_CLOUD = bool(OLLAMA_API_KEY)
_cloud_client = (
    Client(host="https://ollama.com", headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"})
    if _USE_CLOUD else None
)
MCP_SERVER_URL  = f"http://{MCP_HOST}:{MCP_PORT}/mcp"

SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in Confluent and Apache Kafka documentation. "
    "Answer the user's question using ONLY the provided context chunks. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Be concise and accurate."
)


def build_prompt(question: str, context: str) -> str:
    return (
        f"CONTEXT FROM DOCUMENTATION:\n{context}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"Answer based on the context above:"
    )


async def retrieve_context(session: ClientSession, question: str) -> str:
    result = await session.call_tool(
        "confluent_search_documents",
        arguments={"query": question, "top_k": TOP_K},
    )
    return "\n\n".join(b.text for b in result.content if hasattr(b, "text"))


def stream_answer(prompt: str) -> None:
    print("\nGenerating answer...\n" + "-" * 60)
    chat_fn = _cloud_client.chat if _USE_CLOUD else _local_client.chat
    for chunk in chat_fn(
        model=INFERENCE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        stream=True,
    ):
        print(chunk["message"]["content"], end="", flush=True)
    print("\n" + "-" * 60)


async def check_models() -> None:
    # Embedding always runs locally, always check it against local Ollama.
    try:
        response   = _local_client.list()
        models_raw = response["models"] if isinstance(response, dict) else response.models
        available  = [m["name"] if isinstance(m, dict) else m.model for m in models_raw]
        if not any(EMBED_MODEL in a for a in available):
            print(f"Warning: model '{EMBED_MODEL}' not found. Run: ollama pull {EMBED_MODEL}")
    except Exception as e:
        print(f"Warning: cannot connect to Ollama at {OLLAMA_HOST} ({e}). Run: ollama serve")
        return

    # Inference model check depends on where it's actually running.
    if _USE_CLOUD:
        print(f"Using Ollama Cloud for inference: '{INFERENCE_MODEL}'")
        print("(not verified against local models -- will error at call time if the name is wrong)")
    else:
        if not any(INFERENCE_MODEL in a for a in available):
            print(f"Warning: model '{INFERENCE_MODEL}' not found. Run: ollama pull {INFERENCE_MODEL}")


async def rag_loop() -> None:
    await check_models()

    print("RAG system ready (Confluent Docs / Qwen3 / MCP / Qdrant)")
    print(f"Connecting to MCP server at {MCP_SERVER_URL} ...")
    print("(Make sure 'python mcp_server/qdrant_server.py' is running in another terminal)")
    print("Type your question, or 'quit' to exit.\n")

    async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            info = await session.call_tool("confluent_collection_info", arguments={})
            for block in info.content:
                if hasattr(block, "text"):
                    print(f"Knowledge base:\n{block.text}\n")

            while True:
                try:
                    question = input("Question: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye.")
                    break

                if not question:
                    continue
                if question.lower() in ("quit", "exit", "q"):
                    print("Goodbye.")
                    break

                print("\nSearching knowledge base...")
                try:
                    context = await retrieve_context(session, question)

                    if not context or context == "No relevant documents found.":
                        print("No relevant context found.\n")
                        continue

                    print("\nRetrieved chunks:\n" + "=" * 60)
                    print(context)
                    print("=" * 60)

                    stream_answer(build_prompt(question, context))
                    print()
                except Exception as e:
                    print(f"\nSomething went wrong answering that question: {e}")
                    print("You can try asking again.\n")


if __name__ == "__main__":
    asyncio.run(rag_loop())