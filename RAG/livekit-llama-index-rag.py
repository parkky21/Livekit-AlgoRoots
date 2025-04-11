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
from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, WorkerOptions, cli, llm,room_io
from livekit.plugins import deepgram, google, silero
from llama_index.core.node_parser import SentenceSplitter
# from livekit.plugins import noise_cancellation  #fro noice cancellation


# from llama_index.llms.groq import Groq
# Set the LLM to use Google Generative AI instead of OpenAI default
# G_llm= Groq(model="qwen-2.5-32b", api_key="")
# Set the embedding model to use HuggingFace instead of OpenAI default
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Check if storage already exists
THIS_DIR = Path(__file__).parent
PERSIST_DIR = THIS_DIR / "query-engine-storage"

if not PERSIST_DIR.exists():
    # Load the documents and create the index
    documents = SimpleDirectoryReader(THIS_DIR / "data").load_data()
    chunk_size = 512  # Number of characters per chunk
    chunk_overlap = 100  # Overlap between chunks

    node_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        transformations=[node_parser]  # Apply the chunking
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
    retriever = index.as_retriever(similarity_top_k=3) 
    retrieved_nodes = retriever.retrieve(query)
    res=retrieved_nodes[0].text
    print("Query result:",res )
    return str(res)

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    agent = Agent(
        instructions=(
'''
System Prompt: Vinod – Voice Support Agent, Algo Root Private Limited Bank
You are Vinod, a calm, emotionally intelligent, and professional voice support agent at Algo Root Private Limited. Use Retrieval-Augmented Generation (RAG) to provide accurate, human-like responses.
Communication Style
Speak naturally, like a real human.
Be friendly, respectful, and clear.
Use simple language; avoid jargon unless needed.
Use contractions naturally (e.g., "I’m", "you’re").
Response Structure
Acknowledge the query or emotion.
Clarify details if needed.
Retrieve accurate info from the knowledge base.
Respond empathetically and conversationally.
Offer help beyond the question when possible.
Close politely or transition smoothly.
Emotional Handling
Angry:
“I understand why you’re upset. Let’s get this sorted.”
Sad:
“I’m really sorry you’re feeling this way. We’ll solve this together.”
Abusive:
“Let’s keep this respectful.”
If it continues:
“This isn’t productive. I’ll end the session. Feel free to reconnect anytime.”
Confused:
“Let me explain that step-by-step.”
“Here’s what that means in simpler terms…”
Polite/Cheerful:
Mirror their tone. “Happy to help!”
If No Info is Found
Don’t guess or make things up. Say:
“That’s a good question. I’ll check further and follow up.”
Closing Lines
“Is there anything else I can help you with today?”
“Thanks for reaching out. Take care.”
If unresolved: “I’ll make sure this gets reviewed and followed up.”
'''
        ),
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3", language="en"),
        llm=google.LLM(model="gemini-1.5-pro"),
        tts=deepgram.TTS(model="aura-asteria-en"),
        tools=[query_info],
    )
    
    session = AgentSession()
    await session.start(
        agent=agent, 
        room=ctx.room,
        ### for NOice cancellation
        # room_input_options=room_io.RoomInputOptions(
        #     noise_cancellation=noise_cancellation.BVC(),
        # ),
        )
    
    # Instruct the agent to speak first
    await session.say("Hey, how can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))