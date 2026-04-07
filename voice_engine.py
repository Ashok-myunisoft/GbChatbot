"""
voice_engine.py

Handles all voice-specific processing for the /gbaiapi/voice_chat endpoint.
  - Language detection
  - Translation: user language → English (before LLM) and English → user language (after LLM)
  - Response post-processing: strip markdown, shorten for speech
  - Sentence splitting for SSE sentence-by-sentence streaming

Zero impact on existing system — only used by /gbaiapi/voice_chat endpoint.
Requires: pip install deep-translator langdetect
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Optional dependency guards ────────────────────────────────────────────────

try:
    from langdetect import detect as _langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("[VoiceEngine] langdetect not installed. Language detection disabled. Run: pip install langdetect")

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    logger.warning("[VoiceEngine] deep-translator not installed. Translation disabled. Run: pip install deep-translator")


# ── Language detection ────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Detect language of input text.
    Returns ISO 639-1 language code (e.g. 'en', 'ta', 'hi', 'fr').
    Falls back to 'en' if detection fails or package not available.
    Requires minimum 4 words — langdetect is unreliable on shorter text.
    """
    if not LANGDETECT_AVAILABLE or not text.strip():
        return "en"
    # Too short to detect reliably — "Hi", "Okay" misdetect frequently
    if len(text.strip().split()) < 4:
        logger.info(f"[VoiceEngine] Input too short for reliable detection — defaulting to 'en'")
        return "en"
    try:
        lang = _langdetect(text)
        logger.info(f"[VoiceEngine] Detected language: {lang}")
        return lang
    except Exception as e:
        logger.warning(f"[VoiceEngine] Language detection failed: {e}")
        return "en"


# ── Translation ───────────────────────────────────────────────────────────────

_TRANSLATE_CHAR_LIMIT = 4500   # GoogleTranslator hard limit is 5000 — stay safely under


def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate text from source_lang to English.
    Returns original text if already English or translation unavailable.
    Truncates to 4500 chars to stay under GoogleTranslator's 5000 char limit.
    """
    if not TRANSLATOR_AVAILABLE or source_lang == "en" or not text.strip():
        return text
    safe_text = text[:_TRANSLATE_CHAR_LIMIT]
    try:
        translated = GoogleTranslator(source=source_lang, target="en").translate(safe_text)
        logger.info(f"[VoiceEngine] Translated [{source_lang}→en]: {text[:60]} → {(translated or '')[:60]}")
        return translated or text
    except Exception as e:
        logger.warning(f"[VoiceEngine] Translation to English failed: {e} — using original")
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate English response back to the user's original language.
    Returns original text if target is English or translation unavailable.
    Truncates to 4500 chars to stay under GoogleTranslator's 5000 char limit.
    """
    if not TRANSLATOR_AVAILABLE or target_lang == "en" or not text.strip():
        return text
    safe_text = text[:_TRANSLATE_CHAR_LIMIT]
    try:
        translated = GoogleTranslator(source="en", target=target_lang).translate(safe_text)
        logger.info(f"[VoiceEngine] Translated [en→{target_lang}]: {len(text)} chars")
        return translated or text
    except Exception as e:
        logger.warning(f"[VoiceEngine] Translation from English failed: {e} — using English response")
        return text


# ── Response post-processing for voice ───────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove markdown symbols that are not speakable."""
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    # Remove headers (##, ###)
    text = re.sub(r"#{1,6}\s*", "", text)
    # Remove bullet points and dashes at line start
    text = re.sub(r"^\s*[-•*]\s+", "", text, flags=re.MULTILINE)
    # Remove numbered list markers (1. 2. etc)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove excess whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _make_voice_friendly(text: str) -> str:
    """
    Make response concise and speech-friendly.
    - Strips markdown
    - Limits to first 5 sentences to keep voice responses short
    """
    text = _strip_markdown(text)
    sentences = split_into_sentences(text)
    # Keep first 10 sentences for voice — balances completeness vs speech length
    voice_sentences = sentences[:10]
    return " ".join(voice_sentences)


def prepare_response_for_voice(text: str) -> str:
    """
    Full pipeline: strip markdown + make speech-friendly.
    Called after LLM response, before translation back to user language.
    """
    return _make_voice_friendly(text)


# ── Sentence splitting for SSE streaming ─────────────────────────────────────

def split_into_sentences(text: str) -> list:
    """
    Split response text into sentences suitable for streaming one by one.
    Each sentence is a natural speech chunk.
    """
    # Split on sentence-ending punctuation followed by space or newline
    text = text.strip()
    if not text:
        return []
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        # If a sentence is very long (>200 chars), split further at commas
        if len(s) > 200:
            parts = re.split(r'(?<=,)\s+', s)
            sentences.extend([p.strip() for p in parts if p.strip()])
        else:
            sentences.append(s)
    return sentences if sentences else [text]
