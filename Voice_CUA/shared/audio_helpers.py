"""
============================================================================
Voice + CUA — Audio I/O Helpers
============================================================================

PURPOSE:
    Utilities for capturing microphone audio and playing back PCM16 audio
    from Voice Live API responses. Used across all Voice_CUA modules.

============================================================================
"""

import numpy as np
import asyncio
import queue
import threading

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except (ImportError, OSError):
    AUDIO_AVAILABLE = False
    print("  ⚠ sounddevice not available. Audio I/O disabled.")


# Voice Live uses PCM16 at 24kHz by default
SAMPLE_RATE = 24000
CHANNELS = 1
DTYPE = "int16"


class MicrophoneStream:
    """
    Captures microphone audio as PCM16 base64 chunks for Voice Live.

    Usage:
        mic = MicrophoneStream()
        mic.start()
        while running:
            chunk = mic.get_chunk()  # base64-encoded PCM16
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": chunk,
            }))
        mic.stop()
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_duration_ms: int = 100):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self._queue = queue.Queue()
        self._stream = None

    def start(self):
        """Start capturing microphone audio."""
        if not AUDIO_AVAILABLE:
            print("  ⚠ Audio not available. MicrophoneStream inactive.")
            return

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=self.chunk_size,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        """Stop capturing."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self, timeout: float = 0.5) -> str | None:
        """Get the next audio chunk as a base64 string."""
        import base64
        try:
            data = self._queue.get(timeout=timeout)
            return base64.b64encode(data.tobytes()).decode("utf-8")
        except queue.Empty:
            return None

    def _callback(self, indata, frames, time, status):
        """sounddevice callback — push audio into the queue."""
        self._queue.put(indata.copy())


class AudioPlayer:
    """
    Plays PCM16 audio chunks from Voice Live using a ring buffer
    fed into a single continuous OutputStream.

    The key insight: Voice Live sends many small base64 audio deltas
    rapidly. We decode them into a shared byte buffer (protected by a
    lock), and the OutputStream callback reads from that buffer at a
    steady rate. No chunk-level play calls, no clobbering.

    Usage:
        player = AudioPlayer()
        player.start()
        player.play_chunk(base64_audio_data)  # from response.audio.delta
        player.flush()  # wait for buffer to drain
        player.stop()
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._buf = bytearray()  # raw PCM16 bytes
        self._stream = None

    def start(self):
        """Start the continuous audio output stream."""
        if not AUDIO_AVAILABLE:
            return
        if self._stream is not None:
            return
        self._stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=int(self.sample_rate * 0.02),  # 20ms blocks for smooth playback
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        """Stop playback and release the stream."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            self._buf.clear()

    def play_chunk(self, base64_audio: str):
        """Decode a base64 PCM16 delta and append to the playback buffer."""
        import base64
        if not AUDIO_AVAILABLE:
            return
        raw = base64.b64decode(base64_audio)
        with self._lock:
            self._buf.extend(raw)

    def flush(self):
        """Wait for the buffer to drain (up to 5 seconds)."""
        if not AUDIO_AVAILABLE:
            return
        import time
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._lock:
                if len(self._buf) == 0:
                    return
            time.sleep(0.05)

    def clear(self):
        """Clear the buffer immediately (used on barge-in / interruption)."""
        with self._lock:
            self._buf.clear()

    def _callback(self, outdata, frames, time_info, status):
        """RawOutputStream callback — copy bytes from buffer into output."""
        # frames * 2 bytes per sample (int16) * 1 channel
        need = frames * 2 * CHANNELS
        with self._lock:
            available = len(self._buf)
            if available >= need:
                data = bytes(self._buf[:need])
                del self._buf[:need]
            elif available > 0:
                # Partial: use what we have, pad with silence
                data = bytes(self._buf) + b'\x00' * (need - available)
                self._buf.clear()
            else:
                # Nothing: output silence
                data = b'\x00' * need
        outdata[:] = data
