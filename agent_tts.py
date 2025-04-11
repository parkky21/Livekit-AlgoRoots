from dotenv import load_dotenv
load_dotenv()
from livekit import agents
from livekit.agents.voice import AgentSession, Agent, room_io
from livekit.plugins import (
    google,
    silero,
    turn_detector,
)
import whisper
import numpy as np
import logging
from livekit.agents import stt, tts, utils
import asyncio
import torch
from kokoro import KPipeline

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

class KokoroTTS(tts.TTS):
    def __init__(self, default_language="h", lang_to_voice=None):
        # Initialize with fixed sample rate and mono audio
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),  # Change to non-streaming
            sample_rate=24000,
            num_channels=1
        )
        
        self.default_language = default_language
        self.lang_to_voice = lang_to_voice or {"h": "af_heart"}
        self.pipelines = {}
        
        # Pre-initialize pipelines to avoid delays during synthesis
        for lang in self.lang_to_voice:
            try:
                self.pipelines[lang] = KPipeline(lang_code=lang)
            except Exception as e:
                logger.warning(f"Failed to initialize Kokoro pipeline for language {lang}: {e}")

    async def synthesize(self, text: str, *, voice: str | None = None, language: str | None = None) -> tts.SynthesizeStream:
        """Synthesize text into audio using Kokoro TTS."""
        # Determine which language to use
        lang = language or self.default_language
        
        # Get the appropriate voice for the language
        voice_to_use = voice
        if voice_to_use is None and lang in self.lang_to_voice:
            voice_to_use = self.lang_to_voice[lang]
        elif voice_to_use is None:
            voice_to_use = self.lang_to_voice.get(self.default_language, "af_heart")
        
        # Get the appropriate pipeline
        pipeline = self.pipelines.get(lang)
        if pipeline is None:
            # Fall back to default language pipeline if requested language not available
            pipeline = self.pipelines.get(self.default_language, next(iter(self.pipelines.values())))
        
        # Generate audio using Kokoro (simplified non-streaming approach)
        try:
            audio_chunks = []
            generator = pipeline(
                text, 
                voice=voice_to_use, 
                speed=1.0, 
                split_pattern=r'\n+'
            )
            
            for _, _, audio in generator:
                # Convert float audio to int16 PCM format
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_chunks.append(audio_int16.tobytes())
            
            # Concatenate all audio chunks
            audio_data = b''.join(audio_chunks)
            
            # Create an AudioBuffer with the full audio
            buffer = utils.AudioBuffer(
                data=audio_data,
                sample_rate=self.sample_rate,
                num_channels=self.num_channels,
                sample_width=2  # 16-bit PCM = 2 bytes
            )
            
            # Create a simple generator that yields the complete audio once
            async def simple_generator():
                yield tts.SynthesizedAudio(audio=buffer, is_final=True)
            
            return tts.SynthesizeStream(generator=simple_generator())
        
        except Exception as e:
            logger.error(f"Error synthesizing audio with Kokoro TTS: {e}")
            # Return an empty stream in case of error
            async def empty_generator():
                yield tts.SynthesizedAudio(audio=None, is_final=True)
            
            return tts.SynthesizeStream(generator=empty_generator())
        
        
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=WhisperSTT(model_name="tiny", default_language="en"),
        llm=google.LLM(model="gemini-1.5-pro"),
        tts=KokoroTTS(
            default_language="h",
            lang_to_voice={"h": "af_heart", "en": "en_heart"}
        ),
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