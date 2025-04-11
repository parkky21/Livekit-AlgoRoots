from dotenv import load_dotenv

load_dotenv()

from livekit import agents
from livekit.agents.voice import AgentSession, Agent, room_io
from livekit.plugins import (
    google,
    cartesia,
    deepgram,
    silero,
    turn_detector,
)

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en"),
        llm=google.LLM(model="gemini-1.5-pro"),
        tts=deepgram.TTS(model="aura-asteria-en"),
        vad=silero.VAD.load(),
        turn_detection=turn_detector.EOUModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=room_io.RoomInputOptions(
            
        ),
    )

    # Instruct the agent to speak first
    await session.generate_reply()


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))