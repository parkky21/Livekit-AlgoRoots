from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
   
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.plugins import deepgram, google, silero
from llama_index.llms.groq import Groq

# Set the LLM to use Google Generative AI instead of OpenAI default
G_llm= Groq(model="qwen-2.5-32b", api_key="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")

# Set the embedding model to use HuggingFace instead of OpenAI default
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Check if storage already exists
THIS_DIR = Path(__file__).parent
PERSIST_DIR = THIS_DIR / "query-engine-storage"

if not PERSIST_DIR.exists():
    # Load the documents and create the index
    documents = SimpleDirectoryReader(THIS_DIR / "data").load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        )
    # Store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context,
                                    embed_model=embed_model,
                                    )

@llm.function_tool
async def query_info(query: str) -> str:
    """Get more information about a specific topic from the knowledge base"""
    query_engine = index.as_query_engine(use_async=True,llm=G_llm)
    res = await query_engine.aquery(query)
    print("Query result:", res)
    return str(res)

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    agent = Agent(
        instructions=(
            "You are a helpful voice AI assistant. Your interface "
            "with users will be voice. You should use short and concise "
            "responses, and avoiding usage of unpronounceable punctuation. "
            "You can query information from a knowledge base to answer questions."
        ),
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3", language="en"),
        llm=google.LLM(model="gemini-1.5-pro"),
        tts=deepgram.TTS(model="aura-asteria-en"),
        tools=[query_info],
    )
    
    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)
    
    # Instruct the agent to speak first
    await session.say("Hey, how can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))