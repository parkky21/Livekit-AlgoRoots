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
import whisper
import numpy as np
import logging
from livekit.agents import stt, utils

logger = logging.getLogger("voice-agent")

class WhisperSTT(stt.STT):
    def __init__(self, model_name="tiny", default_language="en"):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.model = whisper.load_model(model_name)
        self.default_language = default_language

    async def _recognize_impl(
        self, buffer: utils.AudioBuffer, *, language: str | None = None, conn_options=None
    ) -> stt.SpeechEvent:
        # Merge audio frames into a single buffer
        buffer = utils.merge_frames(buffer)

        # Convert to mono if stereo, and prepare for Whisper (16-bit PCM to float32)
        audio_data = np.frombuffer(buffer.data, dtype=np.int16)
        if buffer.num_channels > 1:
            audio_data = audio_data.reshape(-1, buffer.num_channels).mean(axis=1)
        audio_array = audio_data.astype(np.float32) / 32768.0

        # Use specified language or default
        lang = language or self.default_language

        # Transcribe audio using Whisper
        result = self.model.transcribe(audio_array, language=lang)
        text = result["text"]

        # Return the speech event
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=lang)]
        )

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=WhisperSTT(model_name="tiny",default_language="en"),
        llm=google.LLM(model="gemini-1.5-pro"),
        tts=deepgram.TTS(model="aura-asteria-en"),
        vad=silero.VAD.load(),
        turn_detection=turn_detector.EOUModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=room_io.RoomInputOptions(),
    )

    await session.generate_reply()

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))