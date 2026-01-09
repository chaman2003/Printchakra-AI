"""
Voice AI Module - Whisper + Local LLM Integration
Handles speech-to-text transcription and AI chat responses with GPU-accelerated TTS output

GPU ENHANCEMENTS:
- Whisper: Optimized with FP16, greedy decoding, minimal parameters
- TTS: GPU-accelerated with Coqui TTS (5-10x faster) + CPU fallback
- Memory: Automatic GPU memory management and caching
"""

import logging
import os
import tempfile
import threading
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

from app.config.settings import AI_PROMPT_CONFIG, CONNECTION_CONFIG

# Import chat_flow for unified workflow orchestration
try:
    from app.chat_flow import chat_flow_service, ChatFlowResponse
    CHAT_FLOW_AVAILABLE = True
except ImportError:
    CHAT_FLOW_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("[WARN] Chat flow module not available")

if TYPE_CHECKING:
    from .voice_prompt import VoicePromptManager as VoicePromptManagerType
else:
    VoicePromptManagerType = Any

logger = logging.getLogger(__name__)

OLLAMA_API_TIMEOUT = 60  # Default voice AI timeout; override via config/prompts


# Import Voice AI prompt management
VoicePromptManager: Optional[Any] = None
try:
    from .voice_prompt import VoicePromptManager as _VPM, OLLAMA_API_TIMEOUT as _PM_TIMEOUT
    VoicePromptManager = _VPM
    OLLAMA_API_TIMEOUT = _PM_TIMEOUT
    VOICE_PROMPT_AVAILABLE = True
except ImportError:
    VOICE_PROMPT_AVAILABLE = False
    logger.warning("[WARN] Voice prompt module not available")
except Exception as exc:
    VOICE_PROMPT_AVAILABLE = False
    logger.error(f"[ERROR] Voice prompt module failed to load: {exc}")


def _build_ollama_url(base: str, endpoint: str) -> str:
    """Normalize Ollama endpoint URLs"""

    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    normalized_base = (base or "http://localhost:11434").rstrip("/")
    normalized_endpoint = endpoint or ""
    if not normalized_endpoint.startswith("/"):
        normalized_endpoint = f"/{normalized_endpoint}"
    return f"{normalized_base}{normalized_endpoint}"


_ollama_settings = CONNECTION_CONFIG.get("ollama", {})
_ollama_base = (_ollama_settings.get("base_url") or "http://localhost:11434").rstrip("/")
OLLAMA_TAGS_URL = _build_ollama_url(_ollama_base, _ollama_settings.get("tags_endpoint", "/api/tags"))
OLLAMA_CHAT_URL = _build_ollama_url(_ollama_base, _ollama_settings.get("chat_endpoint", "/api/chat"))

try:
    OLLAMA_TIMEOUT = int(_ollama_settings.get("timeout", OLLAMA_API_TIMEOUT))
except (TypeError, ValueError):
    OLLAMA_TIMEOUT = OLLAMA_API_TIMEOUT

OLLAMA_VERIFY_SSL = bool(_ollama_settings.get("verify_ssl", False))
DEFAULT_VOICE_MODEL = AI_PROMPT_CONFIG.get("default_model", "smollm2:135m")

# Import GPU optimization modules
try:
    from .gpu_optimization import (
        initialize_gpu,
        get_gpu_info,
        get_optimal_device,
        gpu_model_cache,
        gpu_memory_manager,
        log_gpu_memory
    )
    GPU_OPTIMIZATION_AVAILABLE = True
except ImportError:
    GPU_OPTIMIZATION_AVAILABLE = False
    logger.warning("[WARN] GPU optimization module not available")

# Import TTS module
try:
    from .tts_gpu import speak_text_gpu, get_tts_engine_info, _init_tts_engine
    TTS_IMPORTED = True
except ImportError as e:
    TTS_IMPORTED = False
    logger.debug(f"[DEBUG] TTS module import note: {e}")

# Legacy TTS fallback (keep for compatibility)
_tts_engine = None
_tts_lock = threading.Lock()  # Prevent concurrent TTS calls
TTS_AVAILABLE = False

_tts_initialized_once = False  # Track if we've logged TTS init


def truncate_to_word_count(text: str, max_words: int = 20) -> str:
    """
    Truncate text to approximately maximum word count for TTS output,
    but always complete the current sentence to avoid cut-off responses.
    
    Args:
        text: Text to truncate
        max_words: Target maximum number of words (may exceed to complete sentence)
        
    Returns:
        Text truncated at sentence boundary closest to max_words
    """
    if not text or not text.strip():
        return text
    
    words = text.split()
    if len(words) <= max_words:
        return text
    
    # Find sentence boundaries (., !, ?)
    sentence_endings = ['.', '!', '?']
    
    # Build text word by word and track sentence boundaries
    result_words = []
    last_sentence_end_idx = -1
    
    for i, word in enumerate(words):
        result_words.append(word)
        
        # Check if this word ends a sentence
        if any(word.rstrip(')"\'').endswith(end) for end in sentence_endings):
            last_sentence_end_idx = i
            
            # If we've reached or passed max_words, stop at this sentence
            if i + 1 >= max_words:
                break
    
    # If we found a sentence ending, use it
    if last_sentence_end_idx >= 0:
        return " ".join(result_words[:last_sentence_end_idx + 1])
    
    # No sentence ending found - look ahead for the next sentence ending (up to 10 more words)
    max_lookahead = min(len(words), max_words + 10)
    for i in range(max_words, max_lookahead):
        word = words[i]
        if any(word.rstrip(')"\'').endswith(end) for end in sentence_endings):
            return " ".join(words[:i + 1])
    
    # Still no sentence ending - just return at max_words with period
    truncated = " ".join(words[:max_words])
    if not any(truncated.endswith(end) for end in sentence_endings):
        truncated = truncated.rstrip(',;:') + "."
    return truncated


def _init_tts_engine():
    """Initialize TTS engine"""
    global _tts_engine, TTS_AVAILABLE, _tts_initialized_once

    # Use imported TTS module if available
    if TTS_IMPORTED:
        try:
            from .tts_gpu import _init_tts_engine as init_gpu_tts
            init_gpu_tts()
            TTS_AVAILABLE = True
            return True
        except Exception as e:
            logger.debug(f"[DEBUG] TTS initialization: {e}")
    
    # Legacy SAPI5 initialization (fallback)
    if _tts_engine is not None:
        return True

    try:
        import pyttsx3

        engine = pyttsx3.init()
        voices: Any = engine.getProperty("voices")  # type: ignore
        selected_voice: Any = None

        voice_preferences = ["ravi", "david", "zira"]

        for preference in voice_preferences:
            for voice in voices:  # type: ignore
                if preference in voice.name.lower():  # type: ignore
                    selected_voice = voice
                    if not _tts_initialized_once:
                        logger.info(f"[OK] Found preferred voice: {voice.name}")  # type: ignore
                    break
            if selected_voice:
                break

        if not selected_voice and voices:  # type: ignore
            selected_voice = voices[0]  # type: ignore

        if selected_voice:
            engine.setProperty("voice", selected_voice.id)

        engine.setProperty("rate", 200)
        engine.setProperty("volume", 0.9)

        _tts_engine = engine
        TTS_AVAILABLE = True

        if not _tts_initialized_once:
            logger.info("[OK] Text-to-Speech initialized successfully (SAPI5 fallback)")
            _tts_initialized_once = True

        return True

    except Exception as e:
        logger.error(f"[ERROR] TTS initialization failed: {e}")
        TTS_AVAILABLE = False
        return False


def speak_text(text: str) -> bool:
    """
    Speak text using GPU-accelerated TTS with CPU fallback (blocking call)
    
    Uses:
    1. GPU-accelerated Coqui TTS (5-10x faster) if available and GPU detected
    2. Falls back to Windows SAPI5 if GPU unavailable
    
    Args:
        text: Text to speak

    Returns:
        bool: True if speech was successful
    """
    global _tts_lock

    with _tts_lock:
        try:
            # Try GPU TTS first (5-10x faster)
            if TTS_AVAILABLE:
                try:
                    if speak_text_gpu(text):
                        return True
                except Exception as e:
                    logger.debug(f"[DEBUG] GPU TTS failed, falling back to SAPI5: {e}")
            
            # Fallback to legacy SAPI5 implementation
            import gc
            gc.collect()

            import pyttsx3

            engine = pyttsx3.init("sapi5", debug=False)
            voices: Any = engine.getProperty("voices")  # type: ignore
            selected_voice: Any = None

            voice_preferences = ["ravi", "david", "zira"]

            for preference in voice_preferences:
                for voice in voices:  # type: ignore
                    if preference in voice.name.lower():  # type: ignore
                        selected_voice = voice
                        break
                if selected_voice:
                    break

            if not selected_voice and voices:  # type: ignore
                selected_voice = voices[0]  # type: ignore

            if selected_voice:
                engine.setProperty("voice", selected_voice.id)

            engine.setProperty("rate", 220)
            engine.setProperty("volume", 0.9)

            engine.say(text)
            engine.runAndWait()

            try:
                engine.stop()
            except:
                pass

            del engine
            gc.collect()

            return True

        except Exception as e:
            logger.error(f"[ERROR] TTS error: {e}")
            return False


class WhisperTranscriptionService:
    """
    Service for real-time speech transcription using Whisper with GPU acceleration
    
    GPU ENHANCEMENTS:
    - FP16 precision for faster computation
    - Greedy decoding (beam_size=1) for speed
    - Minimal parameters for reduced inference time
    - GPU memory caching and management
    - Model stays in VRAM between requests for faster subsequent transcriptions
    """

    def __init__(
        self, model_path: str = r"C:\Users\chama\OneDrive\Desktop\printchakra\ggml-small-q5_1.bin"
    ):
        """
        Initialize Whisper transcription service with GPU support

        Args:
            model_path: Path to local GGML whisper model file
        """
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.use_whisper_cpp = False
        
        # GPU optimization initialization
        if GPU_OPTIMIZATION_AVAILABLE:
            try:
                initialize_gpu()
                log_gpu_memory("WhisperInit")
            except Exception as e:
                logger.debug(f"[DEBUG] GPU optimization init failed: {e}")

    def _try_load_with_whisper_cpp(self):
        """Try to load using whisper.cpp (if available)"""
        try:
            import whisper_cpp_python  # type: ignore

            if not self.is_loaded:  # Only log on first load
                logger.info(f"Loading GGML model with whisper.cpp: {self.model_path}")
            self.model = whisper_cpp_python.Whisper(self.model_path)
            self.is_loaded = True
            self.use_whisper_cpp = True
            if not hasattr(self, "_logged_load"):  # Only log once
                logger.info("[OK] Whisper GGML model loaded successfully with whisper.cpp")
                self._logged_load = True
            return True
        except Exception as e:
            logger.debug(f"whisper.cpp not available: {e}")
            return False

    def _try_load_with_openai_whisper(self):
        """Fallback to openai-whisper with GPU support - optimized for speed"""
        try:
            import torch
            import whisper  # type: ignore
            from .gpu_optimization import get_optimal_device

            # Use get_optimal_device to prefer NVIDIA GPU
            device = get_optimal_device()  # Returns 'cuda' if available, else 'cpu'

            # Use small model for better accuracy (461MB, good balance)
            if not hasattr(self, "_logged_load"):  # Only log once
                logger.info(f"Loading openai-whisper small model on {device.upper()}")
            self.model = whisper.load_model("small", device=device)
            self.is_loaded = True
            self.use_whisper_cpp = False
            self.device = device
            if not hasattr(self, "_logged_load"):  # Only log once
                logger.info(f"[OK] Whisper small model loaded successfully on {device.upper()}")
                logger.info(f"   Model optimized for accuracy (461MB, better transcription)")
                self._logged_load = True
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to load base model: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())

            # Fallback to base model if small fails
            try:
                logger.info("Falling back to base model...")
                import torch
                import whisper
                from .gpu_optimization import get_optimal_device

                device = get_optimal_device()  # Returns 'cuda' if available, else 'cpu'

                self.model = whisper.load_model("base", device=device)
                self.is_loaded = True
                self.use_whisper_cpp = False
                self.device = device
                if not hasattr(self, "_logged_load"):  # Only log once
                    logger.info(f"[OK] Whisper base model loaded successfully on {device.upper()}")
                    logger.info(f"   Model with balanced performance (244MB)")
                    self._logged_load = True
                return True
            except Exception as fallback_error:
                logger.error(f"[ERROR] Failed to load fallback model: {str(fallback_error)}")
                import traceback

                logger.error(traceback.format_exc())
                return False

    def load_model(self):
        """Load Whisper model into memory"""
        try:
            # Check if GGML model file exists
            if os.path.exists(self.model_path):
                logger.info(f"Found local GGML model: {self.model_path}")

                # Try whisper.cpp first (optimized for GGML)
                if self._try_load_with_whisper_cpp():
                    return True

                # Fallback: Try loading with openai-whisper
                # Note: openai-whisper doesn't directly support GGML, so we fall back to standard model
                logger.info("[OK] Using openai-whisper model on GPU (optimized performance)")
            else:
                logger.info(f"Using standard openai-whisper model on GPU")

            # Use standard model as fallback
            return self._try_load_with_openai_whisper()

        except Exception as e:
            logger.error(f"[ERROR] Failed to load Whisper model: {str(e)}")
            self.is_loaded = False
            return False

    def transcribe_audio(self, audio_data: bytes, language: str = "en") -> Dict[str, Any]:
        """
        Transcribe audio bytes to text

        Args:
            audio_data: Audio bytes in WAV format
            language: Language code (default: en)

        Returns:
            Dict with transcription results and metadata
        """
        if not self.is_loaded:
            if not self.load_model():
                return {"success": False, "error": "Whisper model not loaded", "text": ""}

        try:
            # Validate audio data
            if not audio_data:
                return {"success": False, "error": "Audio data is empty", "text": ""}

            if len(audio_data) < 100:  # Minimum reasonable audio file size
                return {"success": False, "error": "Audio data too small to process", "text": ""}

            # Validate WAV format
            logger.info(f"[INFO] Audio data size: {len(audio_data)} bytes")

            # Check WAV header
            if audio_data[:4] != b"RIFF":
                logger.warning(
                    f"[WARN] Audio doesn't start with RIFF header. First 4 bytes: {audio_data[:4]}"
                )
                # Try to convert or handle non-WAV format
                if audio_data[:4] == b"\x1aE\xdf\xa3":  # WebM signature
                    logger.info("Detected WebM format, attempting conversion...")
                    # FFmpeg will attempt conversion
                elif audio_data[:2] == b"\xff\xfb":  # MP3 signature
                    logger.info("Detected MP3 format, attempting conversion...")
            else:
                logger.info("[OK] Valid RIFF/WAV header detected")

            # Save audio to temporary file with proper extension
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name

            # logger.info(f"Created temp audio file: {temp_audio_path}")

            # Verify file was created and has content
            if not os.path.exists(temp_audio_path):
                return {
                    "success": False,
                    "error": "Failed to save temporary audio file",
                    "text": "",
                }

            file_size = os.path.getsize(temp_audio_path)
            # logger.info(f"Temporary audio file size: {file_size} bytes")

            # Verify file is not corrupted by checking RIFF structure
            with open(temp_audio_path, "rb") as f:
                file_header = f.read(4)
                if file_header != b"RIFF":
                    logger.error(f"[ERROR] Corrupted WAV file! Header is {file_header}, not RIFF")
                    logger.error(f"   First 20 bytes: {audio_data[:20]}")
                    os.unlink(temp_audio_path)
                    return {
                        "success": False,
                        "error": f"Corrupted audio file - invalid WAV header. Got {file_header} instead of RIFF",
                        "text": "",
                    }

            # Transcribe based on model type
            try:
                # Log GPU memory before transcription
                if GPU_OPTIMIZATION_AVAILABLE:
                    log_gpu_memory("TranscribeBefore")
                
                if self.use_whisper_cpp:
                    # Use whisper.cpp for GGML model
                    result: Dict[str, Any] = self.model.transcribe(temp_audio_path)  # type: ignore
                    text = result["result"]
                else:
                    # Use openai-whisper with SPEED-OPTIMIZED GPU settings
                    import torch
                    from .gpu_optimization import get_optimal_device

                    # Check GPU availability for FP16 optimization
                    device = get_optimal_device()  # Returns 'cuda' if available, else 'cpu'
                    use_fp16 = device == 'cuda'

                    # Clear GPU cache before transcription to prevent memory errors
                    if device == 'cuda':
                        try:
                            import torch
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            logger.debug("[DEBUG] GPU cache cleared for transcription")
                        except Exception as e:
                            logger.debug(f"[DEBUG] Could not clear GPU cache: {e}")

                    # Build transcription options with GPU acceleration
                    transcribe_options = {
                        "language": language,
                        "task": "transcribe",
                        "fp16": use_fp16,  # Use FP16 on GPU for 2x speedup
                        "beam_size": 1,  # FAST: Greedy decoding (5→1 = 5x faster)
                        "best_of": 1,  # FAST: Single candidate (5→1 = no extra sampling)
                        "temperature": 0.0,  # FAST: Deterministic, no fallback
                        "compression_ratio_threshold": 2.4,
                        "no_speech_threshold": 0.6,  # Lowered from 0.75 to accept short greetings
                        "logprob_threshold": -1.0,  # Lowered from -0.5 to accept uncertain short speech
                        "condition_on_previous_text": False,  # FAST: Each chunk independent
                        "verbose": False,
                    }
                    
                    try:
                        result: Dict[str, Any] = self.model.transcribe(temp_audio_path, **transcribe_options)  # type: ignore
                    except (RuntimeError, Exception) as te:
                        # CUDA error fallback: Switch to CPU and retry
                        if device == 'cuda' and ('CUDA' in str(te) or 'cuda' in str(te).lower()):
                            logger.warning(f"[WARN] CUDA error during transcription, falling back to CPU: {str(te)[:100]}")
                            try:
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            except:
                                pass
                            
                            # Retry with CPU
                            transcribe_options_cpu = transcribe_options.copy()
                            transcribe_options_cpu['fp16'] = False  # CPU doesn't support FP16
                            result: Dict[str, Any] = self.model.transcribe(temp_audio_path, **transcribe_options_cpu)  # type: ignore
                            logger.info("[OK] Transcription succeeded on CPU fallback")
                        elif isinstance(te, TypeError):
                            # Handle version compatibility issues
                            if "vad_filter" in str(te) or "DecodingOptions" in str(te):
                                logger.warning(f"[WARN] Parameter compatibility issue: {te}")
                                result: Dict[str, Any] = self.model.transcribe(  # type: ignore
                                    temp_audio_path,
                                    language=language,
                                    fp16=use_fp16,
                                    verbose=False
                                )
                            else:
                                raise
                        else:
                            raise
                    
                    text: str = result.get("text", "").strip()  # type: ignore
                    segments: Any = result.get("segments", [])  # type: ignore
                    
                    # Level 1: Check no_speech_prob for each segment (VERY RELAXED)
                    if segments:
                        # Calculate average no_speech probability across all segments
                        avg_no_speech_prob = sum(seg.get("no_speech_prob", 0) for seg in segments) / len(segments)  # type: ignore
                        max_no_speech_prob = max(seg.get("no_speech_prob", 0) for seg in segments)  # type: ignore
                        
                        # RELAXED: Only reject if VERY HIGH confidence it's noise (avg > 0.85 OR max > 0.95)
                        # This allows quiet/distant speech, accents, and background noise during speech
                        if avg_no_speech_prob > 0.85 or max_no_speech_prob > 0.95:
                            logger.warning(f"[WARN] Background noise detected (avg: {avg_no_speech_prob:.2f}, max: {max_no_speech_prob:.2f})")
                            os.unlink(temp_audio_path)
                            return {
                                "success": False,
                                "error": "Only background noise detected. Please speak clearly.",
                                "text": "",
                                "no_speech_detected": True,
                                "auto_retry": True
                            }
                        
                        # Level 2: Check average log probability (confidence in transcription)
                        avg_logprob = sum(seg.get("avg_logprob", -1.0) for seg in segments) / len(segments)  # type: ignore
                        
                        # MUCH MORE RELAXED: Only reject if extremely low confidence (< -1.5)
                        # This accepts normal speech with accents, quiet speech, or some background noise
                        # -1.08 (your case) will now PASS and be transcribed
                        if avg_logprob < -1.5:
                            logger.warning(f"[WARN] Very low confidence transcription (avg_logprob: {avg_logprob:.2f}) - likely background noise")
                            os.unlink(temp_audio_path)
                            return {
                                "success": False,
                                "error": "Unclear audio. Please speak louder and clearer.",
                                "text": "",
                                "no_speech_detected": True,
                                "auto_retry": True
                            }
                        else:
                            # Log acceptance for debugging
                            logger.info(f"[OK] Speech accepted (no_speech: avg={avg_no_speech_prob:.2f}, max={max_no_speech_prob:.2f}, logprob={avg_logprob:.2f})")
                    
                    # Level 3: Check if transcribed text is too short or gibberish (RELAXED)
                    if text:
                        # Filter garbage transcriptions (punctuation-only, digits-only)
                        # Whisper sometimes outputs just "?" or "." or "..." when uncertain
                        import re
                        cleaned_text = re.sub(r'[^\w\s]', '', text).strip()  # Remove all punctuation
                        
                        if not cleaned_text or len(cleaned_text) < 1:
                            logger.warning(f"[WARN] Garbage transcription detected: '{text}' - likely noise, not speech")
                            os.unlink(temp_audio_path)
                            return {
                                "success": False,
                                "error": "Unclear audio. Please speak louder and try again.",
                                "text": "",
                                "no_speech_detected": True,
                                "auto_retry": True
                            }
                        
                        word_count = len(text.split())
                        # RELAXED: Accept even single words (e.g., "print", "scan", "yes", "no")
                        # Only reject completely empty or very suspicious results
                        if word_count < 1:
                            logger.warning(f"[WARN] Transcription too short ({word_count} words): '{text}' - likely background noise")
                            os.unlink(temp_audio_path)
                            return {
                                "success": False,
                                "error": "No clear speech detected. Please try again.",
                                "text": "",
                                "no_speech_detected": True,
                                "auto_retry": True
                            }
                        
                        # Log what we transcribed
                        logger.info(f"[INFO] Transcribed: '{text}' ({word_count} words)")
                    
                    # Level 4: Check for empty or whitespace-only transcription
                    if not text or text.isspace():
                        logger.warning(f"[WARN] Empty transcription - background noise only")
                        os.unlink(temp_audio_path)
                        return {
                            "success": False,
                            "error": "No human speech detected. Please try again.",
                            "text": "",
                            "no_speech_detected": True,
                            "auto_retry": True
                        }
                    
            except Exception as transcribe_error:
                logger.error(f"[ERROR] Whisper transcription failed: {str(transcribe_error)}")
                logger.error(f"   Error type: {type(transcribe_error).__name__}")

                # Try to clear GPU cache and recover
                if GPU_OPTIMIZATION_AVAILABLE:
                    try:
                        from .gpu_optimization import clear_gpu_cache
                        clear_gpu_cache()
                        logger.info("[OK] GPU cache cleared after error")
                    except Exception as cache_error:
                        logger.debug(f"[DEBUG] Could not clear GPU cache: {cache_error}")

                # Check if it's a file access error
                if (
                    "End of file" in str(transcribe_error)
                    or "invalid" in str(transcribe_error).lower()
                ):
                    logger.error(
                        f"   This suggests the audio file is corrupted or not in valid WAV format"
                    )
                    logger.error(f"   File size: {file_size} bytes")

                raise transcribe_error

            # Clean up
            try:
                os.unlink(temp_audio_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to delete temp file immediately: {cleanup_error}")
                # Try to remove it later via garbage collection
                import gc

                gc.collect()
                try:
                    os.unlink(temp_audio_path)
                except:
                    logger.debug(f"Could not delete temp file: {temp_audio_path}")

            # Log GPU memory after transcription and cleanup
            if GPU_OPTIMIZATION_AVAILABLE:
                log_gpu_memory("TranscribeAfter")

            return {
                "success": True,
                "text": text,
                "language": language,
                "segments": result.get("segments", []) if not self.use_whisper_cpp else [],
                "duration": result.get("duration", 0) if not self.use_whisper_cpp else 0,
            }

        except Exception as e:
            logger.error(f"[ERROR] Transcription error: {str(e)}")
            logger.error(f"   Error type: {type(e).__name__}")
            # Try to clean up temp file if it exists
            try:
                if "temp_audio_path" in locals() and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except:
                # Force cleanup with garbage collection
                import gc

                gc.collect()
                try:
                    if "temp_audio_path" in locals() and os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                except:
                    pass
            
            # Cleanup GPU memory on error
            if GPU_OPTIMIZATION_AVAILABLE:
                from .gpu_optimization import clear_gpu_cache
                clear_gpu_cache()
            
            return {"success": False, "error": str(e), "text": ""}


class VoiceChatService:
    """
    Service for AI chat responses using Voice AI via Ollama
    Fast, efficient local inference with intelligent voice command interpretation
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Voice AI chat service with full orchestration awareness

        Args:
            model_name: Ollama model to use
        """
        self.model_name = model_name or DEFAULT_VOICE_MODEL
        self.conversation_history = []
        self.pending_orchestration = None  # Track if waiting for confirmation (print/scan)
        self.ollama_chat_url = OLLAMA_CHAT_URL
        self.ollama_tags_url = OLLAMA_TAGS_URL
        self.api_timeout = max(1, OLLAMA_TIMEOUT) if OLLAMA_TIMEOUT else OLLAMA_API_TIMEOUT
        self.verify_ssl = OLLAMA_VERIFY_SSL
        
        # Import command mappings from centralized module
        if VOICE_PROMPT_AVAILABLE and VoicePromptManager is not None:
            self.command_mappings = VoicePromptManager.get_command_mappings()  # type: ignore
            self.system_prompt = VoicePromptManager.get_system_prompt()  # type: ignore
        else:
            # Fallback if prompt manager not available
            self.command_mappings = {}
            self.system_prompt = ""

    def check_ollama_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            import requests

            timeout = min(5, self.api_timeout or OLLAMA_API_TIMEOUT)
            response = requests.get(
                self.ollama_tags_url,
                timeout=timeout,
                verify=self.verify_ssl,
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                target_fragment = (self.model_name or "").split(":")[0].lower()
                has_model = True
                if target_fragment:
                    has_model = any(target_fragment in (name or "").lower() for name in model_names)
                # Only log once at startup
                if not hasattr(self, "_logged_ollama_check"):
                    logger.info(f"Ollama available via {self.ollama_tags_url}: {has_model}")
                    logger.info(f"Available models: {model_names}")
                    self._logged_ollama_check = True
                return has_model
            return False
        except Exception as e:
            if not hasattr(self, "_logged_ollama_error"):
                logger.error(f"Ollama check failed: {str(e)}")
                self._logged_ollama_error = True
            return False

    def _parse_command_parameters(self, user_message: str, command_type: str) -> Dict[str, Any]:
        """
        Parse parameters from voice command
        
        Args:
            user_message: User's spoken text
            command_type: Detected command type
            
        Returns:
            Dict with parsed parameters
        """
        params = {}
        text_lower = user_message.lower().strip()
        import re
        
        # Parse multiple document selection (e.g., "select first two documents")
        if command_type == "select_multiple_documents":
            # Extract section keywords
            if any(keyword in text_lower for keyword in ["original", "current", "recent"]):
                params["section"] = "current"
            elif "converted" in text_lower:
                params["section"] = "converted"
            elif any(keyword in text_lower for keyword in ["uploaded", "upload", "new"]):
                params["section"] = "upload"
            else:
                params["section"] = "current"
            
            # Check for "select all"
            if "all" in text_lower:
                params["count"] = -1  # -1 means select all
                params["selection_type"] = "all"
                logger.info(f"[PARAM] Multi-doc: select ALL")
                return params
            
            # Word numbers mapping
            word_to_num = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
            }
            
            # Check for "LAST N" patterns first (these are different from first N)
            last_n_match = re.search(r'last\s+(\d+)', text_lower)
            if last_n_match:
                params["count"] = int(last_n_match.group(1))
                params["selection_type"] = "last_n"
                logger.info(f"[PARAM] Multi-doc: last {params['count']} (numeric)")
                return params
            
            # Check for "last word" patterns (last two, last three)
            for word, num in word_to_num.items():
                if f"last {word}" in text_lower:
                    params["count"] = num
                    params["selection_type"] = "last_n"
                    logger.info(f"[PARAM] Multi-doc: last {word} = {num}")
                    return params
            
            # Try to extract count using regex for "first N" patterns (numeric)
            first_n_match = re.search(r'first\s+(\d+)', text_lower)
            if first_n_match:
                params["count"] = int(first_n_match.group(1))
                params["selection_type"] = "first_n"
                logger.info(f"[PARAM] Multi-doc: first {params['count']} (numeric)")
                return params
            
            # Try to extract count using regex for "select N documents" patterns
            select_n_match = re.search(r'select\s+(\d+)\s+documents?', text_lower)
            if select_n_match:
                params["count"] = int(select_n_match.group(1))
                params["selection_type"] = "first_n"
                logger.info(f"[PARAM] Multi-doc: select {params['count']} docs (numeric)")
                return params
            
            # Extract count from word forms: "first two", "first three", etc.
            for word, num in word_to_num.items():
                if f"first {word}" in text_lower:
                    params["count"] = num
                    params["selection_type"] = "first_n"
                    logger.info(f"[PARAM] Multi-doc: first {word} = {num}")
                    return params
                if f"{word} documents" in text_lower or f"{word} document" in text_lower:
                    params["count"] = num
                    params["selection_type"] = "first_n"
                    logger.info(f"[PARAM] Multi-doc: {word} documents = {num}")
                    return params
            
            # Try to parse explicit numbers like "1 and 2" or "1, 2, 3"
            number_matches = re.findall(r'\b(\d+)\b', text_lower)
            if len(number_matches) >= 2:
                params["document_numbers"] = [int(n) for n in number_matches]
                params["count"] = len(params["document_numbers"])
                params["selection_type"] = "specific"
                logger.info(f"[PARAM] Multi-doc: specific docs {params['document_numbers']}")
                return params
            elif len(number_matches) == 1:
                params["count"] = int(number_matches[0])
                params["selection_type"] = "first_n"
                logger.info(f"[PARAM] Multi-doc: single number {params['count']}")
                return params
            
            # Default to 2 if nothing parsed
            params["count"] = 2
            params["selection_type"] = "first_n"
            logger.info(f"[PARAM] Multi-doc: defaulting to 2")
            
            return params
        
        # Parse specific document list selection (e.g., "select documents 4, 6, 8")
        if command_type == "select_specific_documents":
            if any(keyword in text_lower for keyword in ["original", "current", "recent"]):
                params["section"] = "current"
            elif "converted" in text_lower:
                params["section"] = "converted"
            else:
                params["section"] = "current"
            
            # Extract all numbers from the message
            number_matches = re.findall(r'\b(\d+)\b', text_lower)
            if number_matches:
                params["document_numbers"] = [int(n) for n in number_matches]
                params["selection_type"] = "specific"
                logger.info(f"[PARAM] Specific docs: {params['document_numbers']}")
            else:
                params["document_numbers"] = []
                params["selection_type"] = "specific"
            
            return params



        # Parse range document selection (e.g., "select documents 1 to 5", "select from 1 to 5")
        if command_type == "select_document_range":
            # Extract section keywords
            if any(keyword in text_lower for keyword in ["original", "current", "recent"]):
                params["section"] = "current"
            elif "converted" in text_lower:
                params["section"] = "converted"
            else:
                params["section"] = "current"
            
            # Word to number mapping for spoken numbers
            word_to_num = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
                "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5
            }
            
            # Replace word numbers with digits for easier parsing
            processed_text = text_lower
            for word, num in word_to_num.items():
                processed_text = re.sub(rf'\b{word}\b', str(num), processed_text)
            
            # Extract range using various patterns (order matters - most specific first)
            range_patterns = [
                r'from\s+(\d+)\s+to\s+(\d+)',  # from 1 to 5 - PRIORITY
                r'select\s+from\s+(\d+)\s+to\s+(\d+)',  # select from 1 to 5
                r'(\d+)\s+(?:to|through)\s+(\d+)',  # 1 to 5
                r'between\s+(\d+)\s+(?:and|to)\s+(\d+)',  # between 1 and 5
                r'documents?\s+(\d+)\s*(?:to|through|-|–)\s*(\d+)',  # documents 1 to 5, documents 1-5
                r'(\d+)\s*-\s*(\d+)',  # 1-5
            ]
            
            for pattern in range_patterns:
                match = re.search(pattern, processed_text)
                if match:
                    start = int(match.group(1))
                    end = int(match.group(2))
                    params["start"] = min(start, end)
                    params["end"] = max(start, end)
                    params["selection_type"] = "range"
                    logger.info(f"[PARAM] Range selection: {params['start']} to {params['end']}")
                    return params
            
            # Default if no range found
            params["start"] = 1
            params["end"] = 5
            params["selection_type"] = "range"
            logger.info(f"[PARAM] Range selection: defaulting to 1-5")
            return params
        
        # Parse deselect command - mirrors select patterns
        if command_type == "deselect_document":
            if any(keyword in text_lower for keyword in ["original", "current", "recent"]):
                params["section"] = "current"
            elif "converted" in text_lower:
                params["section"] = "converted"
            else:
                params["section"] = "current"
            
            # Word to number mapping
            word_to_num = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
            }
            
            # Check for deselect all FIRST
            if re.search(r'(?:deselect|unselect|clear)\s*(?:all|everything)', text_lower) or text_lower in ["clear selection", "remove selection"]:
                params["deselect_all"] = True
                params["deselect_type"] = "all"
                logger.info(f"[PARAM] Deselect: all")
                return params
            
            # Check for LAST N pattern
            last_match = re.search(r'(?:deselect|unselect|clear)\s+(?:the\s+)?last\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)', text_lower)
            if last_match:
                count_str = last_match.group(1)
                count = word_to_num.get(count_str.lower(), int(count_str) if count_str.isdigit() else 2)
                params["count"] = count
                params["deselect_type"] = "last_n"
                params["deselect_all"] = False
                logger.info(f"[PARAM] Deselect: last {count}")
                return params
            
            # Check for FIRST N pattern
            first_match = re.search(r'(?:deselect|unselect|clear)\s+(?:the\s+)?first\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)', text_lower)
            if first_match:
                count_str = first_match.group(1)
                count = word_to_num.get(count_str.lower(), int(count_str) if count_str.isdigit() else 2)
                params["count"] = count
                params["deselect_type"] = "first_n"
                params["deselect_all"] = False
                logger.info(f"[PARAM] Deselect: first {count}")
                return params
            
            # Check for RANGE pattern (1 to 5)
            range_match = re.search(r'(\d+)\s+(?:to|through)\s+(\d+)', text_lower)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                params["start"] = min(start, end)
                params["end"] = max(start, end)
                params["deselect_type"] = "range"
                params["deselect_all"] = False
                logger.info(f"[PARAM] Deselect: range {params['start']} to {params['end']}")
                return params
            
            # Check for specific document numbers
            number_matches = re.findall(r'\b(\d+)\b', text_lower)
            if number_matches:
                params["document_numbers"] = [int(n) for n in number_matches]
                params["deselect_type"] = "specific"
                params["deselect_all"] = False
                logger.info(f"[PARAM] Deselect: specific {params['document_numbers']}")
                return params
            
            # Default to deselect all if nothing matched
            params["deselect_all"] = True
            params["deselect_type"] = "all"
            logger.info(f"[PARAM] Deselect: defaulting to all")
            return params

        # Parse feed count command (e.g., "feed 5 documents")
        if command_type == "set_feed_count":
            count_match = re.search(r'(\d+)', text_lower)
            if count_match:
                params["count"] = int(count_match.group(1))
            else:
                params["count"] = 1
            logger.info(f"[PARAM] Feed count: {params['count']}")
            return params

        # Parse page selection command (e.g., "odd pages only", "pages 1 to 5")
        if command_type == "set_pages":
            if "odd" in text_lower:
                params["pages"] = "odd"
            elif "even" in text_lower:
                params["pages"] = "even"
            elif "all" in text_lower:
                params["pages"] = "all"
            else:
                # Check for custom range
                range_match = re.search(r'(\d+)\s*(?:to|through|-)\s*(\d+)', text_lower)
                if range_match:
                    params["pages"] = "custom"
                    params["customRange"] = f"{range_match.group(1)}-{range_match.group(2)}"
                else:
                    # Single page
                    single_match = re.search(r'(?:page\s+)?(\d+)(?:\s+only)?', text_lower)
                    if single_match:
                        params["pages"] = "custom"
                        params["customRange"] = single_match.group(1)
                    else:
                        params["pages"] = "all"
            logger.info(f"[PARAM] Pages: {params.get('pages')}, CustomRange: {params.get('customRange')}")
            return params

        # Parse switch_section command
        if command_type == "switch_section":
            if any(keyword in text_lower for keyword in ["converted", "converted files"]):
                params["section"] = "converted"
            elif any(keyword in text_lower for keyword in ["local", "upload", "local files"]):
                params["section"] = "upload"
            elif any(keyword in text_lower for keyword in ["current", "current documents"]):
                params["section"] = "current"
            logger.info(f"[PARAM] Switch section to: {params.get('section')}")
            return params

        # Parse single document selection (e.g., "select original number 2")


        if command_type == "select_document":

            # Extract section keywords and normalize to dashboard terms
            if any(keyword in text_lower for keyword in ["original", "current", "recent"]):
                params["section"] = "current"
            elif "converted" in text_lower:
                params["section"] = "converted"
            elif any(keyword in text_lower for keyword in ["uploaded", "upload", "new"]):
                params["section"] = "upload"
            
            # Extract document number (1-based)
            number_match = re.search(r'(?:number|item|file|doc|#)\s*(\d+)', text_lower)
            if number_match:
                params["document_number"] = int(number_match.group(1))
            else:
                # Try to find standalone number
                number_match = re.search(r'\b(\d+)\b', text_lower)
                if number_match:
                    params["document_number"] = int(number_match.group(1))
            
            if "first" in text_lower:
                params["document_number"] = 1
            elif "second" in text_lower:
                params["document_number"] = 2
            elif "third" in text_lower:
                params["document_number"] = 3
            elif "last" in text_lower:
                params["document_number"] = -1  # indicate special handling downstream
            
            if "section" not in params:
                params["section"] = "current"
            if "document_number" not in params:
                params["document_number"] = 1
        
        # Parse section switching (e.g., "switch to converted")
        elif command_type == "switch_section":
            if any(keyword in text_lower for keyword in ["original", "current", "recent"]):
                params["section"] = "current"
            elif "converted" in text_lower:
                params["section"] = "converted"
            elif any(keyword in text_lower for keyword in ["uploaded", "upload", "new"]):
                params["section"] = "upload"
        
        # Parse color mode settings
        elif command_type == "set_color_mode":
            bw_keywords = ["grayscale", "greyscale", "gray scale", "grey scale", "black and white", 
                          "black & white", "bw", "mono", "monochrome", "no color"]
            color_keywords = ["color", "full color", "in color"]
            
            if any(kw in text_lower for kw in bw_keywords):
                params["color_mode"] = "bw"
            elif any(kw in text_lower for kw in color_keywords):
                params["color_mode"] = "color"
            else:
                params["color_mode"] = "bw"  # Default to bw if detected as color mode command
        
        # Parse quality settings
        elif command_type == "set_quality":
            if any(kw in text_lower for kw in ["draft", "fast", "economy", "low quality", "quick"]):
                params["quality"] = "draft"
            elif any(kw in text_lower for kw in ["best", "premium", "photo", "professional", "excellent", "ultra"]):
                params["quality"] = "professional"
            elif any(kw in text_lower for kw in ["high", "fine", "detailed", "good quality"]):
                params["quality"] = "high"
            else:
                params["quality"] = "normal"
        
        # Parse format settings
        elif command_type == "set_format":
            if "pdf" in text_lower:
                params["format"] = "pdf"
            elif any(kw in text_lower for kw in ["jpeg", "jpg"]):
                params["format"] = "jpeg"
            elif "png" in text_lower:
                params["format"] = "png"
            elif any(kw in text_lower for kw in ["tiff", "tif"]):
                params["format"] = "tiff"
            else:
                params["format"] = "pdf"  # Default
        
        # Parse resolution settings
        elif command_type == "set_resolution":
            dpi_match = re.search(r'(\d+)\s*dpi', text_lower)
            if dpi_match:
                params["resolution"] = dpi_match.group(1)
            elif any(kw in text_lower for kw in ["low", "draft"]):
                params["resolution"] = "150"
            elif any(kw in text_lower for kw in ["high", "fine"]):
                params["resolution"] = "600"
            elif any(kw in text_lower for kw in ["ultra", "maximum"]):
                params["resolution"] = "1200"
            else:
                params["resolution"] = "300"  # Default
        
        # Parse layout settings
        elif command_type == "set_layout":
            if any(kw in text_lower for kw in ["landscape", "horizontal", "wide"]):
                params["layout"] = "landscape"
            elif any(kw in text_lower for kw in ["portrait", "vertical", "tall"]):
                params["layout"] = "portrait"
            else:
                params["layout"] = "portrait"  # Default
        
        # Parse paper size settings
        elif command_type == "set_paper_size":
            if "a4" in text_lower or "a 4" in text_lower:
                params["paper_size"] = "A4"
            elif "a3" in text_lower or "a 3" in text_lower:
                params["paper_size"] = "A3"
            elif "a5" in text_lower or "a 5" in text_lower:
                params["paper_size"] = "A5"
            elif "letter" in text_lower:
                params["paper_size"] = "Letter"
            elif "legal" in text_lower:
                params["paper_size"] = "Legal"
            else:
                params["paper_size"] = "A4"  # Default
        
        # Parse copies settings
        elif command_type == "set_copies":
            copies_match = re.search(r'(\d+)\s*(?:cop(?:y|ies))?', text_lower)
            if copies_match:
                params["copies"] = int(copies_match.group(1))
            else:
                params["copies"] = 1
        
        # Parse duplex settings
        elif command_type == "set_duplex":
            if any(kw in text_lower for kw in ["single", "one side", "front only", "simplex"]):
                params["duplex"] = False
            else:
                params["duplex"] = True  # Default to duplex if command detected
        
        # Parse margins settings
        elif command_type == "set_margins":
            if any(kw in text_lower for kw in ["no margin", "borderless", "none", "zero"]):
                params["margins"] = "none"
            elif any(kw in text_lower for kw in ["narrow", "small", "thin"]):
                params["margins"] = "narrow"
            else:
                params["margins"] = "default"
        
        return params

    def _generate_command_response(self, voice_command: str, command_params: Dict[str, Any], user_message: str) -> str:
        """
        Generate a descriptive response based on the detected command and its parameters.
        This creates unified, contextual responses instead of generic "Processing..." messages.
        
        Args:
            voice_command: The detected command type (e.g., "select_document_range")
            command_params: Parsed parameters for the command
            user_message: Original user message for context
            
        Returns:
            A descriptive string response explaining what action will be taken
        """
        try:
            # Document selection responses
            if voice_command == "select_document_range":
                start = command_params.get("start", 1)
                end = command_params.get("end", 5)
                section = command_params.get("section", "current")
                return f"Selecting documents {start} to {end} from {section} section."
            
            elif voice_command == "select_document":
                doc_num = command_params.get("document_number", 1)
                section = command_params.get("section", "current")
                if doc_num == -1:
                    return f"Selecting the last document from {section} section."
                return f"Selecting document {doc_num} from {section} section."
            
            elif voice_command == "select_multiple_documents":
                selection_type = command_params.get("selection_type", "first_n")
                count = command_params.get("count", 2)
                section = command_params.get("section", "current")
                
                if selection_type == "all" or count == -1:
                    return f"Selecting all documents from {section} section."
                elif selection_type == "last_n":
                    return f"Selecting the last {count} documents from {section} section."
                elif selection_type == "specific":
                    doc_numbers = command_params.get("document_numbers", [])
                    if doc_numbers:
                        nums_str = ", ".join(str(n) for n in doc_numbers)
                        return f"Selecting documents {nums_str} from {section} section."
                    return f"Selecting {count} documents from {section} section."
                else:
                    return f"Selecting the first {count} documents from {section} section."
            
            elif voice_command == "select_specific_documents":
                doc_numbers = command_params.get("document_numbers", [])
                section = command_params.get("section", "current")
                if doc_numbers:
                    nums_str = ", ".join(str(n) for n in doc_numbers)
                    return f"Selecting documents {nums_str} from {section} section."
                return f"Selecting specific documents from {section} section."
            
            # Deselect responses
            elif voice_command == "deselect_document":
                deselect_type = command_params.get("deselect_type", "all")
                section = command_params.get("section", "current")
                
                if command_params.get("deselect_all", False) or deselect_type == "all":
                    return "Clearing all document selections."
                elif deselect_type == "range":
                    start = command_params.get("start", 1)
                    end = command_params.get("end", 5)
                    return f"Deselecting documents {start} to {end}."
                elif deselect_type == "last_n":
                    count = command_params.get("count", 2)
                    return f"Deselecting the last {count} documents."
                elif deselect_type == "first_n":
                    count = command_params.get("count", 2)
                    return f"Deselecting the first {count} documents."
                elif deselect_type == "specific":
                    doc_numbers = command_params.get("document_numbers", [])
                    if doc_numbers:
                        nums_str = ", ".join(str(n) for n in doc_numbers)
                        return f"Deselecting documents {nums_str}."
                return "Deselecting documents."
            
            # Print/scan operation responses
            elif voice_command == "start_print":
                return "Starting print operation with selected documents."
            
            elif voice_command == "start_scan":
                return "Starting scan operation."
            
            elif voice_command == "stop_print":
                return "Stopping current print operation."
            
            elif voice_command == "stop_scan":
                return "Stopping current scan operation."
            
            # Configuration responses
            elif voice_command == "set_color_mode":
                color_mode = command_params.get("color_mode", "color")
                mode_name = "grayscale" if color_mode == "bw" else "color"
                return f"Setting print mode to {mode_name}."
            
            elif voice_command == "set_quality":
                quality = command_params.get("quality", "normal")
                return f"Setting print quality to {quality}."
            
            elif voice_command == "set_paper_size":
                size = command_params.get("paper_size", "A4")
                return f"Setting paper size to {size}."
            
            elif voice_command == "set_copies":
                copies = command_params.get("copies", 1)
                return f"Setting number of copies to {copies}."
            
            elif voice_command == "set_duplex":
                duplex = command_params.get("duplex", False)
                mode = "double-sided" if duplex else "single-sided"
                return f"Setting {mode} printing."
            
            elif voice_command == "set_layout":
                layout = command_params.get("layout", "portrait")
                return f"Setting page layout to {layout}."
            
            elif voice_command == "set_pages":
                pages = command_params.get("pages", "all")
                custom_range = command_params.get("customRange", "")
                if pages == "custom" and custom_range:
                    return f"Setting page range to {custom_range}."
                elif pages == "odd":
                    return "Setting to print odd pages only."
                elif pages == "even":
                    return "Setting to print even pages only."
                return "Setting to print all pages."
            
            elif voice_command == "set_resolution":
                resolution = command_params.get("resolution", "300")
                return f"Setting scan resolution to {resolution} DPI."
            
            elif voice_command == "set_format":
                format_type = command_params.get("format", "pdf")
                return f"Setting output format to {format_type.upper()}."
            
            elif voice_command == "set_margins":
                margins = command_params.get("margins", "default")
                return f"Setting margins to {margins}."
            
            elif voice_command == "set_feed_count":
                count = command_params.get("count", 1)
                return f"Setting document feed count to {count}."
            
            # Navigation responses
            elif voice_command == "switch_section":
                section = command_params.get("section", "current")
                section_names = {"current": "Current Documents", "converted": "Converted Files", "upload": "Uploaded Files"}
                section_name = section_names.get(section, section)
                return f"Switching to {section_name} section."
            
            elif voice_command == "open_settings":
                return "Opening settings panel."
            
            elif voice_command == "close_settings":
                return "Closing settings panel."
            
            elif voice_command == "open_print_config":
                return "Opening print configuration."
            
            elif voice_command == "open_scan_config":
                return "Opening scan configuration."
            
            # Help and info
            elif voice_command == "help":
                return "Here are some commands you can use: 'select document 1', 'select from 1 to 5', 'start print', 'set color mode to grayscale', 'open settings'."
            
            elif voice_command == "status":
                return "Checking current status..."
            
            # Capture operations
            elif voice_command == "capture":
                return "Capturing document image."
            
            elif voice_command == "start_capture":
                return "Starting continuous capture mode."
            
            elif voice_command == "stop_capture":
                return "Stopping capture mode."
            
            # Default fallback
            else:
                # Generate a basic response from the command name
                command_display = voice_command.replace("_", " ").title()
                return f"Processing: {command_display}."
                
        except Exception as e:
            logger.error(f"Error generating command response: {e}")
            return f"Processing your command: {user_message}"


    def interpret_voice_command(self, user_message: str) -> tuple[Optional[str], float]:
        """
        Intelligently interpret voice commands with fuzzy matching
        
        Args:
            user_message: User's spoken text
            
        Returns:
            Tuple of (command_type, confidence_score) or (None, 0) if no match
        """
        try:
            from difflib import SequenceMatcher
            import re
            
            user_lower = user_message.lower().strip()
            best_command = None
            best_confidence = 0.0
            
            # PRIORITY CHECK: Detect multi-document selection patterns FIRST
            # This prevents "select first 2 documents" from matching "select_document"
            multi_doc_patterns = [
                r'select\s+(?:the\s+)?first\s+(\d+|two|three|four|five|six|seven|eight|nine|ten)\s*(?:documents?|files?)?',
                r'(?:first|last)\s+(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+documents?',
                r'select\s+(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+documents?',
                r'select\s+all\s*(?:documents?|files?)?',
                r'all\s+documents?',
                r'choose\s+all',
                r'select\s+(?:the\s+)?last\s+(\d+|two|three|four|five)',
            ]
            
            for pattern in multi_doc_patterns:
                if re.search(pattern, user_lower):
                    logger.info(f"[PRIORITY] MULTI-DOCUMENT PATTERN: '{user_message}' matches pattern '{pattern}'")
                    return "select_multiple_documents", 0.95
            
            # PRIORITY CHECK: Detect DESELECT patterns BEFORE select patterns
            # This prevents "deselect 4" from matching "select 4"
            deselect_priority_patterns = [
                r'\bdeselect\b',  # Any command containing "deselect"
                r'\bunselect\b',  # Any command containing "unselect"
                r'\bclear\s+(?:document|doc|file)?\s*\d+',  # "clear 4", "clear document 4"
                r'\bclear\s+(?:the\s+)?(?:first|last)',  # "clear first 2", "clear last 3"
                r'\bclear\s+all',  # "clear all"
                r'\bremove\s+(?:document|doc|file)?\s*\d+',  # "remove 4", "remove document 4"
            ]
            for pattern in deselect_priority_patterns:
                if re.search(pattern, user_lower):
                    logger.info(f"[PRIORITY] DESELECT PATTERN: '{user_message}' matches pattern '{pattern}'")
                    return "deselect_document", 0.95

            # PRIORITY CHECK: Detect SINGLE DOCUMENT selection patterns (select document 7)

            single_doc_patterns = [
                r'select\s+(?:document|doc|file)\s+(\d+)$',  # "select document 7"
                r'(?:select|pick|choose)\s+(?:document|doc|file)?\s*#?(\d+)$',  # "select #7" or "select 7"
                r'(?:document|doc|file)\s+(\d+)$',  # "document 7"
            ]
            for pattern in single_doc_patterns:
                if re.search(pattern, user_lower):
                    logger.info(f"[PRIORITY] SINGLE DOCUMENT PATTERN: '{user_message}' matches pattern '{pattern}'")
                    return "select_document", 0.95
            
            # PRIORITY CHECK: Detect DESELECT patterns BEFORE range (so "deselect 2 to 5" works)
            deselect_patterns = [
                # Deselect last N / first N
                r'(?:deselect|unselect|clear)\s+(?:the\s+)?last\s+(\d+|two|three|four|five)',
                r'(?:deselect|unselect|clear)\s+(?:the\s+)?first\s+(\d+|two|three|four|five)',
                # Deselect range (1 to 5)
                r'(?:deselect|unselect|clear)\s+(?:documents?\s+)?(\d+)\s+(?:to|through)\s+(\d+)',
                r'(?:deselect|unselect|clear)\s+from\s+(\d+)\s+to\s+(\d+)',
                # Deselect single document - explicit patterns
                r'deselect\s+(?:document|doc|file)?\s*(\d+)$',  # "deselect 5", "deselect document 5"
                r'deselect\s+(\d+)$',  # "deselect 5" - explicit
                # Deselect specific documents (multiple)
                r'(?:deselect|unselect|clear)\s+(?:documents?|files?)?\s*(\d+(?:\s*(?:,|and)\s*\d+)+)',
                # Deselect all
                r'(?:deselect|unselect|clear)\s*(?:all|everything)',
                r'clear\s+selection',
                r'remove\s+selection',
            ]

            
            for pattern in deselect_patterns:
                if re.search(pattern, user_lower):
                    logger.info(f"[PRIORITY] DESELECT PATTERN: '{user_message}' matches pattern '{pattern}'")
                    return "deselect_document", 0.95
            
            # PRIORITY CHECK: Detect RANGE selection patterns (1 to 5, between, from X to Y)

            range_patterns = [
                r'select\s+from\s+(\d+)\s+to\s+(\d+)',  # select from 1 to 5 - NEW PRIORITY
                r'from\s+(\d+)\s+to\s+(\d+)',  # from 1 to 5 - PRIORITY
                r'(?:select\s+)?(?:documents?\s+)?(\d+)\s+(?:to|through)\s+(\d+)',  # 1 to 5, documents 1 to 5
                r'between\s+(?:document\s+)?(\d+)\s+(?:and|to)\s+(\d+)',  # between 1 and 5
                r'(?:select\s+)?documents?\s+(\d+)\s*-\s*(\d+)',  # documents 1-5
            ]
            
            for pattern in range_patterns:
                if re.search(pattern, user_lower):
                    logger.info(f"[PRIORITY] RANGE SELECTION PATTERN: '{user_message}' matches pattern '{pattern}'")
                    return "select_document_range", 0.95

            
            # PRIORITY CHECK: Detect SPECIFIC DOCUMENT LIST patterns (select documents 4, 6, 8)
            specific_list_patterns = [
                r'select\s+(?:documents?|files?)\s+(\d+(?:\s*(?:,|and)\s*\d+)+)',  # select documents 4, 6, 8
                r'select\s+(\d+)\s*,\s*(\d+)',  # select 4, 6
                r'documents?\s+(\d+)\s*(?:,|and)\s*(\d+)',  # documents 4 and 6
            ]
            
            for pattern in specific_list_patterns:
                if re.search(pattern, user_lower):
                    logger.info(f"[PRIORITY] SPECIFIC LIST PATTERN: '{user_message}' matches pattern '{pattern}'")
                    return "select_specific_documents", 0.95
            

            # PRIORITY CHECK: Detect UNDO patterns
            undo_patterns = [
                r'\bundo\b', r'\brevert\b', r'go\s+back', r'oops'
            ]
            for pattern in undo_patterns:
                if re.search(pattern, user_lower):
                    logger.info(f"[PRIORITY] UNDO PATTERN: '{user_message}' matches pattern '{pattern}'")
                    return "undo_action", 0.95

            # PRIORITY CHECK: Detect PROCEED patterns (context-aware navigation)
            proceed_patterns = [
                r'\bproceed\b', r'\bcontinue\b', r'next\s+step', r'go\s+ahead',
                r'move\s+on', r'confirm\s+selection', r'done\s+selecting',
                r"that's\s+all", r'\bready\b', r'proceed\s+to\s+next',
                r'let\'?s\s+go', r'\bstart\b', r'\bbegin\b'
            ]
            for pattern in proceed_patterns:
                if re.search(pattern, user_lower):
                    logger.info(f"[PRIORITY] PROCEED PATTERN: '{user_message}' matches pattern '{pattern}'")
                    return "proceed_action", 0.95

            # PRIORITY CHECK: Detect SWITCH SECTION patterns
            switch_section_patterns = [
                r'switch\s+to\s+(?:converted|converted\s+files?)',  # "switch to converted files"
                r'switch\s+to\s+(?:local|upload|local\s+files?)',  # "switch to local files"
                r'switch\s+to\s+(?:current|current\s+documents?)',  # "switch to current documents"
                r'(?:show|go\s+to|open)\s+(?:converted|converted\s+files?)',  # "show converted files"
                r'(?:show|go\s+to|open)\s+(?:upload|local\s+files?)',  # "show upload"
                r'(?:show|go\s+to|open)\s+(?:current|current\s+documents?)',  # "show current documents"
            ]
            for pattern in switch_section_patterns:
                if re.search(pattern, user_lower):
                    logger.info(f"[PRIORITY] SWITCH SECTION PATTERN: '{user_message}' matches pattern '{pattern}'")
                    return "switch_section", 0.95

            # PRIORITY CHECK: Detect FEED COUNT patterns (e.g., "feed 5 documents")

            feed_patterns = [
                r'feed\s+(\d+)\s*(?:documents?|pages?)?',
                r'(?:set\s+)?feed\s+count\s+(?:to\s+)?(\d+)',
            ]
            for pattern in feed_patterns:
                if re.search(pattern, user_lower):
                    logger.info(f"[PRIORITY] FEED COUNT PATTERN: '{user_message}' matches pattern '{pattern}'")
                    return "set_feed_count", 0.95

            # PRIORITY CHECK: Detect PAGE SELECTION patterns
            page_patterns = [
                r'odd\s+pages?\s*(?:only)?',
                r'even\s+pages?\s*(?:only)?',
                r'(?:custom\s+)?pages?\s+(\d+)\s+(?:to|through)\s+(\d+)',
                r'(?:custom\s+)?page\s+(\d+)\s*(?:only)?',
                r'all\s+pages',
            ]
            for pattern in page_patterns:
                if re.search(pattern, user_lower):
                    logger.info(f"[PRIORITY] PAGE SELECTION PATTERN: '{user_message}' matches pattern '{pattern}'")
                    return "set_pages", 0.95


            # Check each command type
            for command_type, keywords in self.command_mappings.items():
                for keyword in keywords:
                    # Exact match gets highest confidence
                    if user_lower == keyword:
                        logger.info(f"[OK] EXACT MATCH: '{user_message}' → {command_type}")
                        return command_type, 1.0
                    
                    # Substring match within message (contains keyword)
                    if keyword in user_lower:
                        confidence = 0.9
                        if confidence > best_confidence:
                            best_command = command_type
                            best_confidence = confidence
                        logger.info(f"[MATCH] SUBSTRING MATCH: '{user_message}' contains '{keyword}' → {command_type} (confidence: {confidence})")
                        continue
                    
                    # Fuzzy match (similarity score)
                    similarity = SequenceMatcher(None, user_lower, keyword).ratio()
                    if similarity > 0.75:  # >75% similar
                        if similarity > best_confidence:
                            best_command = command_type
                            best_confidence = similarity
                        logger.info(f"🔀 FUZZY MATCH: '{user_message}' ≈ '{keyword}' → {command_type} (confidence: {similarity:.2f})")
            
            if best_command and best_confidence > 0.6:
                if best_command == "select_document":
                    required = any(keyword in user_lower for keyword in ["document", "file", "doc", "page", "item"])
                    if not required:
                        logger.info(
                            f"[SKIP] Select command lacked document context → '{user_message}'"
                        )
                        return None, 0.0
                if best_command == "switch_section":
                    required = any(keyword in user_lower for keyword in ["section", "tab", "converted", "original", "upload"])
                    if not required:
                        logger.info(
                            f"[SKIP] Switch section command lacked section keywords → '{user_message}'"
                        )
                        return None, 0.0

                logger.info(f"[OK] COMMAND INTERPRETED: {best_command} (confidence: {best_confidence:.2f})")
                return best_command, best_confidence
            
            logger.info(f"[NO MATCH] NO COMMAND MATCH: '{user_message}' (best match: {best_command} @ {best_confidence:.2f})")
            return None, 0.0
            
        except Exception as e:
            logger.error(f"Error interpreting voice command: {e}")
            return None, 0.0

    def generate_response(self, user_message: str) -> Dict[str, Any]:
        """
        Generate AI response to user message with workflow orchestration.
        
        WORKFLOW:
        1. Greeting → Fixed introduction
        2. "print"/"scan" → Ask for confirmation ("say yes to proceed")
        3. "yes" with pending → Trigger orchestration
        4. Anything else → Ask Ollama AI
        
        Args:
            user_message: User's text input
            
        Returns:
            Dict with AI response and metadata
        """
        try:
            import requests

            user_lower = user_message.lower().strip()
            
            logger.info(f"[AI] Processing: '{user_message}' | Pending: {self.pending_orchestration}")

            # ===== STEP 1: GREETING =====
            # Check if this is a greeting and respond with introduction
            greeting_words = ["hello", "hi", "hey", "greetings", "howdy", "good morning", "good afternoon", "good evening", "hola"]
            is_greeting = any(word in user_lower for word in greeting_words) or user_lower in greeting_words
            
            if is_greeting and not self.pending_orchestration:
                greeting_response = "Hi there! I am PrintChakra AI, your voice-controlled document assistant. I can help you print or scan documents. Just say 'print' or 'scan' to get started."
                logger.info(f"[GREETING] Recognized greeting: '{user_message}'")
                
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": greeting_response})
                
                return {
                    "success": True,
                    "response": greeting_response,
                    "tts_response": greeting_response,
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "tts_enabled": TTS_AVAILABLE,
                    "spoken": False,
                }

            # ===== STEP 2: PRINT/SCAN COMMAND → ASK CONFIRMATION =====
            print_keywords = ["print", "printing", "i want to print", "print document", "print documents"]
            scan_keywords = ["scan", "scanning", "i want to scan", "scan document", "scan documents"]
            
            is_print_command = any(keyword in user_lower for keyword in print_keywords) and "?" not in user_message
            is_scan_command = any(keyword in user_lower for keyword in scan_keywords) and "?" not in user_message
            
            if is_print_command and not self.pending_orchestration:
                self.pending_orchestration = "print"
                ai_response = "Do you want to print? Say yes to proceed."
                logger.info(f"[PRINT] Print command detected - asking for confirmation")
                
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": ai_response})
                
                return {
                    "success": True,
                    "response": ai_response,
                    "tts_response": ai_response,
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "tts_enabled": TTS_AVAILABLE,
                    "spoken": False,
                    "awaiting_confirmation": True,
                    "pending_mode": "print",
                }
            
            if is_scan_command and not self.pending_orchestration:
                self.pending_orchestration = "scan"
                ai_response = "Do you want to scan? Say yes to proceed."
                logger.info(f"[SCAN] Scan command detected - asking for confirmation")
                
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": ai_response})
                
                return {
                    "success": True,
                    "response": ai_response,
                    "tts_response": ai_response,
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "tts_enabled": TTS_AVAILABLE,
                    "spoken": False,
                    "awaiting_confirmation": True,
                    "pending_mode": "scan",
                }

            # ===== STEP 3: CONFIRMATION → TRIGGER ORCHESTRATION =====
            if self.pending_orchestration:
                confirmation_words = ["yes", "proceed", "go ahead", "okay", "ok", "sure", "yep", "yeah", "ye", "confirm", "do it"]
                is_confirmation = any(user_lower == word or user_lower.startswith(word + " ") for word in confirmation_words)
                
                if is_confirmation:
                    mode = self.pending_orchestration
                    self.pending_orchestration = None  # Clear pending state
                    
                    ai_response = f"Opening {mode} interface now!"
                    logger.info(f"[OK] CONFIRMATION RECEIVED - TRIGGERING ORCHESTRATION: {mode}")
                    
                    self.conversation_history.append({"role": "user", "content": user_message})
                    self.conversation_history.append({"role": "assistant", "content": ai_response})
                    
                    return {
                        "success": True,
                        "response": ai_response,
                        "tts_response": ai_response,
                        "model": self.model_name,
                        "timestamp": datetime.now().isoformat(),
                        "tts_enabled": TTS_AVAILABLE,
                        "spoken": False,
                        "orchestration_trigger": True,
                        "orchestration_mode": mode,
                    }
                else:
                    # User said something else - clear pending and continue
                    logger.info(f"[WARN] User said '{user_message}' instead of confirmation, clearing pending")
                    self.pending_orchestration = None

            # ===== STEP 3.5: SETTINGS COMMANDS (landscape, portrait, color, copies, etc.) =====
            # Check for voice commands and settings BEFORE sending to Ollama
            voice_command, command_confidence = self.interpret_voice_command(user_message)
            
            # Check for multi-settings command (e.g., "landscape with 3 copies in color")
            multi_settings = self._parse_multi_settings_command(user_message)
            
            if multi_settings.get("has_settings"):
                settings = multi_settings.get("settings", {})
                response_text = multi_settings.get("response", "Settings updated.")
                tts_text = multi_settings.get("tts_response", response_text)
                
                logger.info(f"[SETTINGS] Detected settings command: {settings}")
                
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                # Determine voice command based on first setting type
                detected_command = None
                command_params = settings.copy()
                
                if "layout" in settings:
                    detected_command = "set_layout"
                elif "colorMode" in settings or "color_mode" in settings:
                    detected_command = "set_color_mode"
                elif "copies" in settings:
                    detected_command = "set_copies"
                elif "resolution" in settings:
                    detected_command = "set_resolution"
                elif "paper_size" in settings or "paperSize" in settings:
                    detected_command = "set_paper_size"
                elif "quality" in settings:
                    detected_command = "set_quality"
                elif "duplex" in settings:
                    detected_command = "set_duplex"
                elif "format" in settings:
                    detected_command = "set_format"
                
                return {
                    "success": True,
                    "response": response_text,
                    "tts_response": tts_text,
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "tts_enabled": TTS_AVAILABLE,
                    "spoken": False,
                    "voice_command": detected_command,
                    "command_params": command_params,
                    "settings_updated": True,
                    "config_updates": settings,
                }
            
            # Check for single voice command (document selection, navigation, etc.)
            if voice_command and command_confidence >= 0.8:
                command_params = self._parse_command_parameters(user_message, voice_command)
                
                # Generate DESCRIPTIVE response for the command using actual params
                response_text = self._generate_command_response(voice_command, command_params, user_message)
                logger.info(f"[VOICE_CMD] Detected command: {voice_command} with params: {command_params}")
                
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                return {
                    "success": True,
                    "response": response_text,
                    "tts_response": response_text,
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "tts_enabled": TTS_AVAILABLE,
                    "spoken": False,
                    "voice_command": voice_command,
                    "command_params": command_params,
                }

            # ===== STEP 4: GENERAL CONVERSATION → ASK OLLAMA =====
            self.conversation_history.append({"role": "user", "content": user_message})

            # Build messages for Ollama
            messages = [
                {"role": "system", "content": self.system_prompt}
            ] + self.conversation_history

            # Call Ollama API
            if VOICE_PROMPT_AVAILABLE and VoicePromptManager is not None:
                query = VoicePromptManager.build_ollama_query(self.model_name, messages)  # type: ignore
            else:
                query = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": 100,
                        "num_ctx": 1024,
                        "repeat_penalty": 1.2,
                        "stop": ["\n\n", "User:", "Assistant:"],
                    },
                }
            
            response = requests.post(
                self.ollama_chat_url,
                json=query,
                timeout=self.api_timeout or OLLAMA_API_TIMEOUT,
                verify=self.verify_ssl,
            )

            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("message", {}).get("content", "").strip()

                # Format response if prompt manager available
                if VOICE_PROMPT_AVAILABLE and VoicePromptManager is not None:
                    ai_response = VoicePromptManager.format_response(ai_response)  # type: ignore
                else:
                    # Basic cleanup
                    ai_response = ai_response.replace("**", "").replace("*", "")
                    import re
                    ai_response = re.sub(r'\s+', ' ', ai_response).strip()
                    if ai_response and ai_response[-1] not in ".!?":
                        ai_response += "."

                # Add assistant response to history
                self.conversation_history.append({"role": "assistant", "content": ai_response})

                # Keep only last 8 exchanges (16 messages) for context
                if len(self.conversation_history) > 16:
                    self.conversation_history = self.conversation_history[-16:]

                logger.info(f"[AI] Response: {ai_response[:100]}...")

                return {
                    "success": True,
                    "response": ai_response,
                    "tts_response": ai_response,
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "tts_enabled": TTS_AVAILABLE,
                    "spoken": False,
                }
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return {
                    "success": False,
                    "error": f"Ollama API error: {response.status_code}",
                    "response": "",
                }

        except Exception as e:
            logger.error(f"[ERROR] Chat generation error: {str(e)}")
            return {"success": False, "error": str(e), "response": ""}

    def _parse_multi_settings_command(self, user_message: str) -> Dict[str, Any]:
        """
        Parse multiple settings from a single voice input (e.g., "print in landscape with grayscale at 300 dpi")
        and generate a consolidated response.
        
        Args:
            user_message: User's voice input
            
        Returns:
            Dict with detected settings and a combined response describing all changes
        """
        text_lower = user_message.lower().strip()
        
        # Extract all settings from the message
        all_settings = self._extract_config_parameters(text_lower)
        
        if not all_settings:
            return {"has_settings": False, "settings": {}, "response": None, "changes": []}
        
        # Also check for settings commands using _parse_command_parameters for each recognized command type
        settings_command_types = [
            "set_color_mode", "set_quality", "set_format", "set_resolution",
            "set_layout", "set_paper_size", "set_copies", "set_duplex", "set_margins"
        ]
        
        for cmd_type in settings_command_types:
            cmd_params = self._parse_command_parameters(user_message, cmd_type)
            if cmd_params:
                all_settings.update(cmd_params)
        
        # Build a list of human-readable changes
        changes = []
        settings_map = {
            "colorMode": ("Color mode", lambda v: "Black & White" if v == "bw" else "Color"),
            "color_mode": ("Color mode", lambda v: "Black & White" if v == "bw" else "Color"),
            "layout": ("Layout", lambda v: v.capitalize()),
            "paperSize": ("Paper size", str),
            "paper_size": ("Paper size", str),
            "resolution": ("Resolution", lambda v: f"{v} DPI"),
            "quality": ("Quality", lambda v: v.capitalize()),
            "copies": ("Copies", str),
            "duplex": ("Double-sided", lambda v: "On" if v else "Off"),
            "margins": ("Margins", lambda v: v.capitalize()),
            "format": ("Format", lambda v: v.upper()),
            "scanTextMode": ("OCR", lambda v: "Enabled" if v else "Disabled"),
            "scanMode": ("Scan mode", lambda v: "Multi-page" if v == "multi" else "Single page"),
            "pages": ("Pages", lambda v: v.capitalize()),
            "pagesPerSheet": ("Pages per sheet", str),
            "scale": ("Scale", lambda v: f"{v}%"),
        }
        
        for key, value in all_settings.items():
            if key in settings_map:
                label, formatter = settings_map[key]
                changes.append(f"{label}: {formatter(value)}")
            elif not key.endswith("Custom") and not key.endswith("Range"):
                # Include other settings with basic formatting
                changes.append(f"{key}: {value}")
        
        # Generate response
        if len(changes) == 0:
            return {"has_settings": False, "settings": all_settings, "response": None, "changes": []}
        elif len(changes) == 1:
            response = f"Done! {changes[0]}."
        elif len(changes) == 2:
            response = f"Done! {changes[0]} and {changes[1]}."
        else:
            response = f"Done! Updated {len(changes)} settings: " + ", ".join(changes[:-1]) + f", and {changes[-1]}."
        
        logger.info(f"[MULTI-SETTINGS] Parsed {len(changes)} settings from: '{user_message}'")
        logger.info(f"[MULTI-SETTINGS] Changes: {changes}")
        
        return {
            "has_settings": True,
            "settings": all_settings,
            "response": response,
            "changes": changes,
            "tts_response": f"Updated {len(changes)} settings." if len(changes) > 2 else response
        }

    def _extract_config_parameters(self, text: str) -> Dict[str, Any]:
        """
        Extract configuration parameters from user text
        
        Args:
            text: User command text (lowercase)
            
        Returns:
            Dict with extracted parameters
        """
        params: Dict[str, Any] = {}
        import re

        def contains_any(phrases: Any) -> bool:
            return any(phrase in text for phrase in phrases)

        # Color / mono detection - check BW first since color is more generic
        bw_keywords = ["black and white", "black & white", "bw", "mono", "monochrome", 
                       "greyscale", "gray scale", "grey scale", "grayscale", "gray mode", "no color"]
        color_keywords = ["full color", "color copy", "print in color", "in color", "with color"]
        
        if contains_any(bw_keywords):
            params["colorMode"] = "bw"
        elif contains_any(color_keywords) or "color" in text:
            params["colorMode"] = "color"


        # Layout detection - include common typos from speech recognition
        landscape_variants = ["landscape", "landscsape", "landcape", "lanscape", "landcsape", 
                             "horizontal", "wide", "sideways", "land scape", "lands cape"]
        portrait_variants = ["portrait", "portait", "portraite", "vertical", "tall", 
                            "upright", "normal orientation"]
        
        if contains_any(landscape_variants):
            params["layout"] = "landscape"
        elif contains_any(portrait_variants):
            params["layout"] = "portrait"

        # Resolution detection
        dpi_match = re.search(r'(\d+)\s*dpi', text)
        if dpi_match:
            params["resolution"] = dpi_match.group(1)
        elif contains_any(["ultra quality", "high quality", "high res", "best quality"]):
            params["resolution"] = "600"
            params["quality"] = "high"
        elif contains_any(["draft quality", "fast draft", "low quality", "low res"]):
            params["resolution"] = "150"
            params["quality"] = "draft"

        # Copies detection (for print)
        copies_match = re.search(r'(\d+)\s*(?:cop(?:y|ies)|prints|pages)', text)
        if copies_match:
            params["copies"] = int(copies_match.group(1))
        elif contains_any(["single copy", "one copy"]):
            params["copies"] = 1

        # Paper size detection
        if contains_any(["a4", "a 4"]):
            params["paperSize"] = "A4"
        elif contains_any(["letter size", "us letter", "letter paper"]):
            params["paperSize"] = "Letter"
        elif contains_any(["legal size", "legal paper"]):
            params["paperSize"] = "Legal"
        elif "a3" in text or "a 3" in text:
            params["paperSize"] = "A3"

        # Page range detection (print)
        page_range_match = re.search(r'page(?:s)?\s+(\d+)(?:\s*-\s*|\s+to\s+)(\d+)', text)
        if page_range_match:
            params["pages"] = "custom"
            params["customRange"] = f"{page_range_match.group(1)}-{page_range_match.group(2)}"
        elif contains_any(["odd pages", "odd page", "only odd", "odd only", "just odd"]):
            params["pages"] = "odd"
        elif contains_any(["even pages", "even page", "only even", "even only", "just even"]):
            params["pages"] = "even"
        elif "all pages" in text or "entire document" in text:
            params["pages"] = "all"

        # Pages per sheet detection
        pages_per_sheet_match = re.search(r'(\d+)\s*(?:per\s*(?:sheet|page|side))', text)
        if pages_per_sheet_match:
            params["pagesPerSheet"] = pages_per_sheet_match.group(1)

        # Scale detection
        scale_match = re.search(r'(\d{2,3})\s*(?:%|percent)', text)
        if scale_match:
            params["scale"] = int(scale_match.group(1))

        # Margin detection
        if contains_any(["no margin", "borderless", "edge to edge", "full bleed"]):
            params["margins"] = "none"
        elif contains_any(["narrow margin", "small margin", "thin margin"]):
            params["margins"] = "narrow"
        elif contains_any(["default margin", "standard margin", "normal margin"]):
            params["margins"] = "default"

        # Duplex / simplex detection
        if contains_any(["double sided", "two sided", "both sides", "duplex"]):
            params["duplex"] = True
        elif contains_any(["single sided", "one sided", "front only", "simplex"]):
            params["duplex"] = False

        # Quality detection (explicit phrases)
        if contains_any(["high quality", "best quality", "premium quality"]):
            params["quality"] = "high"
        elif contains_any(["normal quality", "standard quality"]):
            params["quality"] = "normal"
        elif contains_any(["draft quality", "eco mode", "economy mode"]):
            params["quality"] = "draft"

        # Text mode detection (for scan)
        if contains_any(["text mode", "ocr", "extract text", "enable ocr", "turn on ocr"]):
            params["scanTextMode"] = True
        elif contains_any(["disable ocr", "turn off ocr", "no text mode"]):
            params["scanTextMode"] = False

        # Format detection (scan/export)
        if "pdf" in text:
            params["format"] = "pdf"
        elif "png" in text:
            params["format"] = "png"
        elif "tiff" in text or "tif" in text:
            params["format"] = "tiff"
        elif "jpg" in text or "jpeg" in text:
            params["format"] = "jpg"

        # Scan-specific page mode detection
        scan_page_match = re.search(r'scan\s+page(?:s)?\s+(\d+)(?:\s*-\s*|\s+to\s+)(\d+)', text)
        if scan_page_match:
            params["scanPageMode"] = "custom"
            params["scanCustomRange"] = f"{scan_page_match.group(1)}-{scan_page_match.group(2)}"
        elif contains_any(["scan odd", "scan only odd"]):
            params["scanPageMode"] = "odd"
        elif contains_any(["scan even", "scan only even"]):
            params["scanPageMode"] = "even"
        elif contains_any(["scan everything", "scan all pages"]):
            params["scanPageMode"] = "all"

        # Scan mode (single vs multi)
        if contains_any(["batch scan", "multi page", "multiple pages", "continuous scan", "stack feed"]):
            params["scanMode"] = "multi"
        elif contains_any(["single page", "one page", "single scan"]):
            params["scanMode"] = "single"

        # Orientation hints that explicitly mention scan
        if contains_any(["scan landscape"]):
            params["scanLayout"] = "landscape"
        elif contains_any(["scan portrait"]):
            params["scanLayout"] = "portrait"

        # Paper size hints mentioning scan specifically
        if contains_any(["scan letter", "scan on letter"]):
            params["scanPaperSize"] = "Letter"
        elif contains_any(["scan legal"]):
            params["scanPaperSize"] = "Legal"

        return params

    def reset_conversation(self):
        """Clear conversation history and pending orchestration"""
        self.conversation_history = []
        self.pending_orchestration = None
        logger.info("Conversation history cleared")


class VoiceAIOrchestrator:
    """
    Orchestrates the complete voice AI workflow:
    Audio → Whisper → Text → Voice AI → Response
    """

    def __init__(self):
        """Initialize voice AI orchestrator"""
        self.whisper_service = WhisperTranscriptionService()
        self.chat_service = VoiceChatService()
        self.session_active = False
        
        # Use chat_flow_service for unified workflow orchestration
        self.use_chat_flow = CHAT_FLOW_AVAILABLE
        if self.use_chat_flow:
            self.chat_flow_service = chat_flow_service
            logger.info("[OK] Voice AI using unified ChatFlowService for workflow orchestration")

        # Initialize TTS in background thread (non-blocking to prevent COM hang)
        def init_tts_bg():
            try:
                _init_tts_engine()
            except Exception as e:
                logger.debug(f"[DEBUG] TTS background init: {e}")
        
        tts_thread = threading.Thread(target=init_tts_bg, daemon=True)
        tts_thread.start()

    def start_session(self) -> Dict[str, Any]:
        """Start a new voice AI session"""
        try:
            # Load Whisper model
            if not self.whisper_service.is_loaded:
                whisper_loaded = self.whisper_service.load_model()
                if not whisper_loaded:
                    return {"success": False, "error": "Failed to load Whisper model"}

            # Check Ollama
            ollama_available = self.chat_service.check_ollama_available()
            if not ollama_available:
                return {
                    "success": False,
                    "error": "Ollama not available or smollm2:135m model not found",
                }

            # Reset conversation
            self.chat_service.reset_conversation()
            self.session_active = True
            
            # Start chat flow session if available
            if self.use_chat_flow:
                self.chat_flow_service.start_session()
                logger.info("[OK] ChatFlow session started with voice AI")

            # logger.info("[OK] Voice AI session started")
            return {
                "success": True,
                "message": "Voice AI session started",
                "whisper_loaded": True,
                "ollama_available": True,
            }

        except Exception as e:
            logger.error(f"[ERROR] Session start error: {str(e)}")
            return {"success": False, "error": str(e)}

    def process_voice_input(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Process voice input through complete pipeline
        Requires "hey" wake word to trigger AI processing

        Args:
            audio_data: Audio bytes (WAV format)

        Returns:
            Dict with transcription and AI response
        """
        if not self.session_active:
            return {"success": False, "error": "No active session. Start a session first."}

        try:
            # Step 1: Transcribe audio
            transcription = self.whisper_service.transcribe_audio(audio_data)

            if not transcription.get("success"):
                return {
                    "success": False,
                    "error": f"Transcription failed: {transcription.get('error')}",
                    "stage": "transcription",
                    "requires_keyword": True,
                }

            user_text = transcription.get("text", "").strip()

            if not user_text:
                return {
                    "success": False,
                    "error": "No speech detected - please speak clearly",
                    "stage": "transcription",
                    "requires_keyword": True,
                }

            logger.info(f"[INFO] Transcribed text: {user_text}")

            # Convert to lowercase for processing
            user_text_lower = user_text.lower()

            # Filter out only very obvious filler speech / noise to prevent unwanted processing
            # BUT allow greetings, single words, and natural conversation phrases to pass through
            # Removed "thank you", "thanks", "thank" - these are natural conversation
            filler_phrases = [
                "hmm", "umm", "uh", "huh", "hm", "um",  # Thinking sounds only
                "yeah yeah", "yep yep", "uh huh",  # Repeated sounds
            ]
            
            # Greetings that should be processed by AI (not filtered as filler)
            greeting_words = ["hi", "hey", "hello", "howdy", "hiya", "yo"]
            
            user_text_stripped = user_text_lower.strip().strip(".,!?")
            
            # Check if it's a greeting - if so, let it through to the AI
            is_greeting = user_text_stripped in greeting_words
            
            # Check if entire input is just obvious filler noise (but NOT a greeting)
            # REMOVED: len(user_text_stripped) < 2 - allow single words/letters
            if not is_greeting and user_text_stripped in filler_phrases:
                logger.info(f"[SKIP] Filler noise detected, ignoring: '{user_text}'")
                return {
                    "success": True,
                    "user_text": user_text,
                    "ai_response": "Listening...",
                    "filler_speech_detected": True,
                    "auto_retry": True,  # Signal frontend to continue listening
                    "session_ended": False,
                    "requires_keyword": False,
                }

            # Remove optional wake words from beginning (if present)
            # But preserve the original text for context
            original_user_text = user_text
            wake_words = ["hey", "hi", "hello", "okay"]

            for wake_word in wake_words:
                if user_text_lower.startswith(wake_word):
                    # Remove wake word from beginning
                    remaining_text = user_text[len(wake_word):].strip()
                    # Only remove if there's more content after the wake word
                    # If wake word IS the entire message, keep it for greeting handling
                    if remaining_text:
                        user_text = remaining_text
                        user_text_lower = user_text.lower()
                        logger.info(f"[OK] Removed wake word, processing: {user_text}")
                    else:
                        # Wake word is the entire message - treat as greeting
                        logger.info(f"[OK] Wake word '{wake_word}' is greeting, keeping original")
                    break

            # Process all speech input (no wake word required)
            logger.info(f"[OK] Processing speech: {user_text}")

            # Check for exit keyword
            if "bye printchakra" in user_text_lower or "goodbye" in user_text_lower:
                self.session_active = False
                # End chat flow session if active
                if self.use_chat_flow:
                    self.chat_flow_service.end_session()
                return {
                    "success": True,
                    "user_text": user_text,
                    "ai_response": "Goodbye! Voice session ended.",
                    "session_ended": True,
                    "requires_keyword": False,
                }

            # Step 2: Process through ChatFlowService for workflow orchestration
            # Fall back to VoiceChatService for general conversation
            chat_response = None
            
            if self.use_chat_flow and self.chat_flow_service.session_active:
                try:
                    # Use ChatFlowService for workflow commands
                    flow_response: ChatFlowResponse = self.chat_flow_service.process_text(user_text)
                    
                    # Convert ChatFlowResponse to dict format expected by voice module
                    chat_response = {
                        "success": flow_response.success,
                        "response": flow_response.ai_response,
                        "tts_response": flow_response.tts_response or flow_response.ai_response,
                        "voice_command": flow_response.voice_command,
                        "command_params": flow_response.command_params or {},
                        "model": "chat_flow",
                        "timestamp": datetime.now().isoformat(),
                        "tts_enabled": TTS_AVAILABLE,
                        "spoken": False,
                        # Forward ChatFlow specific fields
                        "current_step": flow_response.current_step.value,
                        "open_document_selector": flow_response.open_document_selector,
                        "open_config_panel": flow_response.open_config_panel,
                        "open_review_panel": flow_response.open_review_panel,
                        "execute_job": flow_response.execute_job,
                        "config_updates": flow_response.config_updates,
                        "interaction_mode": flow_response.interaction_mode,
                    }
                    
                    logger.info(f"[CHAT_FLOW] Processed via ChatFlowService: {flow_response.current_step} | Mode: {flow_response.interaction_mode}")
                    
                except Exception as e:
                    logger.error(f"[ERROR] ChatFlow processing failed: {e}, falling back to VoiceChatService")
                    chat_response = None
            
            # Fall back to VoiceChatService if ChatFlow not available or failed
            if not chat_response:
                chat_response = self.chat_service.generate_response(user_text)

            if not chat_response.get("success"):
                return {
                    "success": False,
                    "error": f"Chat generation failed: {chat_response.get('error')}",
                    "user_text": user_text,
                    "stage": "chat",
                    "requires_keyword": False,
                }

            ai_response = chat_response.get("response", "")
            # TTS should speak exactly what's displayed in the chat
            # Use truncate_to_word_count to keep it reasonable length but complete sentences
            ai_response = truncate_to_word_count(ai_response, max_words=25)
            tts_response = chat_response.get("tts_response", ai_response)  # Use tts_response from chat_flow if available
            
            # If tts_response wasn't provided, sync it with ai_response
            if not tts_response or tts_response == ai_response:
                tts_response = ai_response  # TTS says exactly what's displayed
            
            voice_command = chat_response.get("voice_command")
            command_params = chat_response.get("command_params", {})
            
            # Forward ChatFlow specific fields
            current_step = chat_response.get("current_step")
            open_document_selector = chat_response.get("open_document_selector", False)
            open_config_panel = chat_response.get("open_config_panel", False)
            open_review_panel = chat_response.get("open_review_panel", False)
            execute_job = chat_response.get("execute_job", False)
            config_updates = chat_response.get("config_updates")
            
            # IMPORTANT: Forward awaiting_confirmation and pending_mode from chat service
            awaiting_confirmation = chat_response.get("awaiting_confirmation", False)
            pending_mode = chat_response.get("pending_mode")

            # Step 3: Check for orchestration triggers in AI response (legacy support)
            orchestration_trigger = chat_response.get("orchestration_trigger")
            orchestration_mode = chat_response.get("orchestration_mode")
            
            if not orchestration_trigger and "TRIGGER_ORCHESTRATION:" in ai_response:
                # Extract orchestration mode from trigger
                trigger_start = ai_response.index("TRIGGER_ORCHESTRATION:")
                trigger_end = ai_response.find(" ", trigger_start)
                if trigger_end == -1:
                    trigger_end = len(ai_response)
                
                trigger_text = ai_response[trigger_start:trigger_end]
                if "print" in trigger_text.lower():
                    orchestration_mode = "print"
                    orchestration_trigger = True
                elif "scan" in trigger_text.lower():
                    orchestration_mode = "scan"
                    orchestration_trigger = True
                
                # Remove trigger from response (clean display text)
                ai_response = ai_response.replace(trigger_text, "").strip()
                # Keep tts_response in sync - TTS says what's displayed
                ai_response = truncate_to_word_count(ai_response, max_words=25)
                tts_response = ai_response
            
            # Step 4: Extract configuration parameters from user text
            config_params = self._extract_config_parameters(user_text_lower)

            # FINAL SYNC: Ensure tts_response always equals ai_response
            tts_response = ai_response

            return {
                "success": True,
                "user_text": user_text,
                "full_text": original_user_text,  # Include original for system message
                "ai_response": ai_response,
                "tts_response": tts_response,
                "voice_command": voice_command,
                "command_params": command_params,
                "transcription_language": transcription.get("language"),
                "model": chat_response.get("model"),
                "session_ended": False,
                "requires_keyword": False,
                # ChatFlow workflow fields
                "current_step": current_step,
                "open_document_selector": open_document_selector,
                "open_config_panel": open_config_panel,
                "open_review_panel": open_review_panel,
                "execute_job": execute_job,
                "config_updates": config_updates,
                # Legacy orchestration fields
                "orchestration_trigger": orchestration_trigger,
                "orchestration_mode": orchestration_mode,
                "config_params": config_params,
                "awaiting_confirmation": awaiting_confirmation,  # Forward confirmation state
                "pending_mode": pending_mode,  # Forward pending mode for confirmation
            }

        except Exception as e:
            logger.error(f"[ERROR] Voice input processing error: {str(e)}")
            return {"success": False, "error": str(e), "stage": "unknown", "requires_keyword": True}

    def _extract_config_parameters(self, text: str) -> Dict[str, Any]:
        """
        Extract configuration parameters from user text
        
        Args:
            text: User command text (lowercase)
            
        Returns:
            Dict with extracted parameters
        """
        params: Dict[str, Any] = {}
        import re

        def contains_any(phrases):
            return any(phrase in text for phrase in phrases)

        # Color / mono detection - check BW first since color is more generic
        bw_keywords = ["black and white", "black & white", "bw", "mono", "monochrome", 
                       "greyscale", "gray scale", "grey scale", "grayscale", "gray mode", "no color"]
        color_keywords = ["full color", "color copy", "print in color", "in color", "with color"]
        
        if contains_any(bw_keywords):
            params["colorMode"] = "bw"
        elif contains_any(color_keywords) or "color" in text:
            params["colorMode"] = "color"


        # Layout detection - include common typos from speech recognition
        landscape_variants = ["landscape", "landscsape", "landcape", "lanscape", "landcsape", 
                             "horizontal", "wide", "sideways", "land scape", "lands cape"]
        portrait_variants = ["portrait", "portait", "portraite", "vertical", "tall", 
                            "upright", "normal orientation"]
        
        if contains_any(landscape_variants):
            params["layout"] = "landscape"
        elif contains_any(portrait_variants):
            params["layout"] = "portrait"

        # Resolution detection
        dpi_match = re.search(r'(\d+)\s*dpi', text)
        if dpi_match:
            params["resolution"] = dpi_match.group(1)
        elif contains_any(["ultra quality", "high quality", "high res", "best quality"]):
            params["resolution"] = "600"
            params["quality"] = "high"
        elif contains_any(["draft quality", "fast draft", "low quality", "low res"]):
            params["resolution"] = "150"
            params["quality"] = "draft"

        # Copies detection (for print)
        copies_match = re.search(r'(\d+)\s*(?:cop(?:y|ies)|prints|pages)', text)
        if copies_match:
            params["copies"] = int(copies_match.group(1))
        elif contains_any(["single copy", "one copy"]):
            params["copies"] = 1

        # Paper size detection
        if contains_any(["a4", "a 4"]):
            params["paperSize"] = "A4"
        elif contains_any(["letter size", "us letter", "letter paper"]):
            params["paperSize"] = "Letter"
        elif contains_any(["legal size", "legal paper"]):
            params["paperSize"] = "Legal"
        elif "a3" in text or "a 3" in text:
            params["paperSize"] = "A3"

        # Page range detection (print)
        page_range_match = re.search(r'page(?:s)?\s+(\d+)(?:\s*-\s*|\s+to\s+)(\d+)', text)
        if page_range_match:
            params["pages"] = "custom"
            params["customRange"] = f"{page_range_match.group(1)}-{page_range_match.group(2)}"
        elif contains_any(["odd pages", "odd page", "only odd", "odd only", "just odd"]):
            params["pages"] = "odd"
        elif contains_any(["even pages", "even page", "only even", "even only", "just even"]):
            params["pages"] = "even"
        elif "all pages" in text or "entire document" in text:
            params["pages"] = "all"

        # Pages per sheet detection
        pages_per_sheet_match = re.search(r'(\d+)\s*(?:per\s*(?:sheet|page|side))', text)
        if pages_per_sheet_match:
            params["pagesPerSheet"] = pages_per_sheet_match.group(1)

        # Scale detection
        scale_match = re.search(r'(\d{2,3})\s*(?:%|percent)', text)
        if scale_match:
            params["scale"] = int(scale_match.group(1))

        # Margin detection
        if contains_any(["no margin", "borderless", "edge to edge", "full bleed"]):
            params["margins"] = "none"
        elif contains_any(["narrow margin", "small margin", "thin margin"]):
            params["margins"] = "narrow"
        elif contains_any(["default margin", "standard margin", "normal margin"]):
            params["margins"] = "default"

        # Duplex / simplex detection
        if contains_any(["double sided", "two sided", "both sides", "duplex"]):
            params["duplex"] = True
        elif contains_any(["single sided", "one sided", "front only", "simplex"]):
            params["duplex"] = False

        # Quality detection (explicit phrases)
        if contains_any(["high quality", "best quality", "premium quality"]):
            params["quality"] = "high"
        elif contains_any(["normal quality", "standard quality"]):
            params["quality"] = "normal"
        elif contains_any(["draft quality", "eco mode", "economy mode"]):
            params["quality"] = "draft"

        # Text mode detection (for scan)
        if contains_any(["text mode", "ocr", "extract text", "enable ocr", "turn on ocr"]):
            params["scanTextMode"] = True
        elif contains_any(["disable ocr", "turn off ocr", "no text mode"]):
            params["scanTextMode"] = False

        # Format detection (scan/export)
        if "pdf" in text:
            params["format"] = "pdf"
        elif "png" in text:
            params["format"] = "png"
        elif "tiff" in text or "tif" in text:
            params["format"] = "tiff"
        elif "jpg" in text or "jpeg" in text:
            params["format"] = "jpg"

        # Scan-specific page mode detection
        scan_page_match = re.search(r'scan\s+page(?:s)?\s+(\d+)(?:\s*-\s*|\s+to\s+)(\d+)', text)
        if scan_page_match:
            params["scanPageMode"] = "custom"
            params["scanCustomRange"] = f"{scan_page_match.group(1)}-{scan_page_match.group(2)}"
        elif contains_any(["scan odd", "scan only odd"]):
            params["scanPageMode"] = "odd"
        elif contains_any(["scan even", "scan only even"]):
            params["scanPageMode"] = "even"
        elif contains_any(["scan everything", "scan all pages"]):
            params["scanPageMode"] = "all"

        # Scan mode (single vs multi)
        if contains_any(["batch scan", "multi page", "multiple pages", "continuous scan", "stack feed"]):
            params["scanMode"] = "multi"
        elif contains_any(["single page", "one page", "single scan"]):
            params["scanMode"] = "single"

        # Orientation hints that explicitly mention scan
        if contains_any(["scan landscape"]):
            params["scanLayout"] = "landscape"
        elif contains_any(["scan portrait"]):
            params["scanLayout"] = "portrait"

        # Paper size hints mentioning scan specifically
        if contains_any(["scan letter", "scan on letter"]):
            params["scanPaperSize"] = "Letter"
        elif contains_any(["scan legal"]):
            params["scanPaperSize"] = "Legal"

        return params

    def end_session(self) -> Dict[str, Any]:
        """End voice AI session"""
        self.session_active = False
        self.chat_service.reset_conversation()
        
        # End chat flow session if active
        if self.use_chat_flow and self.chat_flow_service.session_active:
            self.chat_flow_service.end_session()
            logger.info("ChatFlow session ended with voice AI")
        
        logger.info("Voice AI session ended")
        return {"success": True, "message": "Voice AI session ended"}

    def speak_text_response(self, text: str) -> Dict[str, Any]:
        """
        Speak text using TTS (blocking call)
        Used to play TTS after message is displayed

        Args:
            text: Text to speak

        Returns:
            dict: Status of TTS operation
        """
        try:
            if not TTS_AVAILABLE:
                return {"success": False, "error": "TTS not available"}

            # logger.info("🔊 Starting TTS (blocking)...")
            speak_success = speak_text(text)

            if speak_success:
                # logger.info("[OK] TTS completed successfully")
                return {"success": True, "spoken": True}
            else:
                logger.warning("[WARN] TTS returned False")
                return {"success": False, "error": "TTS failed to speak"}

        except Exception as e:
            logger.error(f"[ERROR] TTS error: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}


# Global orchestrator instance
voice_ai_orchestrator = VoiceAIOrchestrator()


def transcribe_audio_file(audio_path: str) -> Dict[str, Any]:
    """
    Transcribe an audio file to text.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dict with transcription results
    """
    try:
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        transcriber = WhisperTranscriptionService()
        return transcriber.transcribe_audio(audio_data)
    except Exception as e:
        logger.error(f"[ERROR] Transcription error: {e}")
        return {"success": False, "error": str(e), "text": ""}
