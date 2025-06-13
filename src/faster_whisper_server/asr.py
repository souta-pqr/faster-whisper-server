from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from faster_whisper_server.api_models import TranscriptionSegment, TranscriptionWord
from faster_whisper_server.text_utils import Transcription

if TYPE_CHECKING:
    from faster_whisper import transcribe

    from faster_whisper_server.audio import Audio

logger = logging.getLogger(__name__)


class FasterWhisperASR:
    def __init__(
        self,
        whisper: transcribe.WhisperModel,
        **kwargs,
    ) -> None:
        self.whisper = whisper
        self.transcribe_opts = kwargs

    def _transcribe(
        self,
        audio: Audio,
        prompt: str | None = None,
        beam_size: int | None = None,
    ) -> tuple[Transcription, transcribe.TranscriptionInfo]:
        start = time.perf_counter()
        
        # beam_sizeが指定されている場合は設定に追加
        transcribe_kwargs = self.transcribe_opts.copy()
        if beam_size is not None:
            transcribe_kwargs['beam_size'] = beam_size
            
        segments, transcription_info = self.whisper.transcribe(
            audio.data,
            initial_prompt=prompt,
            word_timestamps=True,
            **transcribe_kwargs,
        )
        segments = TranscriptionSegment.from_faster_whisper_segments(segments)
        words = TranscriptionWord.from_segments(segments)
        for word in words:
            word.offset(audio.start)
        transcription = Transcription(words)
        end = time.perf_counter()
        logger.info(
            f"Transcribed {audio} in {end - start:.2f} seconds. Prompt: {prompt}. Beam size: {beam_size}. Transcription: {transcription.text}"
        )
        return (transcription, transcription_info)

    async def transcribe(
        self,
        audio: Audio,
        prompt: str | None = None,
        beam_size: int | None = None,
    ) -> tuple[Transcription, transcribe.TranscriptionInfo]:
        """Wrapper around _transcribe so it can be used in async context."""
        # is this the optimal way to execute a blocking call in an async context?
        # TODO: verify performance when running inference on a CPU
        return await asyncio.get_running_loop().run_in_executor(
            None,
            self._transcribe,
            audio,
            prompt,
            beam_size,
        )