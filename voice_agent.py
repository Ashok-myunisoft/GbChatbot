"""
voice_agent.py

Voice helper layer for audio upload, speech-to-text, text-to-speech,
and temporary audio file serving.

This module is intentionally self-contained so it can be reused without
changing the rest of the chatbot pipeline.
"""

import asyncio
import io
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from openai import OpenAI

import voice_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gbaiapi", tags=["Voice AI"])

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
STT_MODEL = os.getenv("STT_MODEL", "whisper-1")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
AUDIO_TEMP_PATH = os.getenv("AUDIO_TEMP_PATH", "./temp_audio")
MAX_AUDIO_SIZE = os.getenv("MAX_AUDIO_SIZE", "25MB")
VOICE_REQUEST_TIMEOUT = float(os.getenv("VOICE_REQUEST_TIMEOUT", "120"))

_openai_client: Optional[OpenAI] = None
_LIVE_SESSIONS: dict[str, dict] = {}


def _parse_size(size_value: str) -> int:
    value = (size_value or "").strip().upper()
    if not value:
        return 25 * 1024 * 1024
    if value.endswith("MB"):
        return int(float(value[:-2].strip()) * 1024 * 1024)
    if value.endswith("M"):
        return int(float(value[:-1].strip()) * 1024 * 1024)
    if value.endswith("KB"):
        return int(float(value[:-2].strip()) * 1024)
    if value.endswith("K"):
        return int(float(value[:-1].strip()) * 1024)
    return int(float(value))


MAX_AUDIO_SIZE_BYTES = _parse_size(MAX_AUDIO_SIZE)


def get_openai_client() -> OpenAI:
    """Return a cached OpenAI client."""
    global _openai_client
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def ensure_audio_dir() -> Path:
    path = Path(AUDIO_TEMP_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_old_audio_files(max_age_seconds: int = 24 * 3600, max_files: int = 200) -> None:
    """Remove stale temporary audio files."""
    audio_dir = ensure_audio_dir()
    now = time.time()

    files = []
    for item in audio_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".mp3":
            try:
                stat = item.stat()
                files.append((stat.st_mtime, item))
            except OSError:
                continue

    for mtime, item in files:
        if now - mtime > max_age_seconds:
            try:
                item.unlink()
            except OSError:
                logger.warning(f"[VoiceAgent] Failed to remove stale audio file: {item}")

    if len(files) > max_files:
        files.sort(key=lambda x: x[0])
        for _, item in files[: len(files) - max_files]:
            try:
                item.unlink()
            except OSError:
                logger.warning(f"[VoiceAgent] Failed to remove excess audio file: {item}")


def _validate_audio_extension(filename: str) -> bool:
    ext = Path(filename or "").suffix.lower()
    return ext in {".mp3", ".wav", ".webm", ".m4a", ".mp4", ".mpeg", ".ogg"}


async def transcribe_audio_bytes(audio_bytes: bytes, filename: str = "voice_input.wav") -> str:
    """
    Transcribe audio bytes using OpenAI Whisper.
    Supports mp3, wav, webm, m4a and related browser audio containers.
    """
    if not audio_bytes:
        return ""

    if not _validate_audio_extension(filename):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    def _do_transcribe() -> str:
        client = get_openai_client()
        file_obj = io.BytesIO(audio_bytes)
        file_obj.name = filename
        result = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=file_obj,
        )
        text = getattr(result, "text", None) or str(result)
        return text.strip()

    return await asyncio.to_thread(_do_transcribe)


def _write_tts_audio(text: str, output_path: Path) -> None:
    client = get_openai_client()
    speech = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
    )

    if hasattr(speech, "stream_to_file"):
        speech.stream_to_file(str(output_path))
        return

    if hasattr(speech, "write_to_file"):
        speech.write_to_file(str(output_path))
        return

    content = getattr(speech, "content", None)
    if content is None and hasattr(speech, "read"):
        content = speech.read()

    if hasattr(content, "read"):
        content = content.read()

    if not isinstance(content, (bytes, bytearray)):
        raise RuntimeError("Unexpected OpenAI TTS response format.")

    with open(output_path, "wb") as f:
        f.write(content)


async def synthesize_speech(text: str) -> str:
    """Generate MP3 audio from text and return the audio filename."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Cannot synthesize empty text.")

    audio_dir = ensure_audio_dir()
    audio_name = f"voice_{uuid.uuid4().hex}.mp3"
    output_path = audio_dir / audio_name

    await asyncio.to_thread(_write_tts_audio, cleaned, output_path)
    return audio_name


def build_audio_url(audio_name: str) -> str:
    return f"/gbaiapi/voice_audio/{audio_name}"


@router.get("/voice_audio/{audio_name}")
async def serve_voice_audio(audio_name: str):
    """Serve a temporary generated MP3 file."""
    safe_name = os.path.basename(audio_name)
    audio_path = ensure_audio_dir() / safe_name
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(str(audio_path), media_type="audio/mpeg", filename=safe_name)


async def clean_and_speak_response(raw_response: str, target_lang: str) -> tuple[str, str, str]:
    """
    Convert chatbot output into speech-friendly localized text and audio.
    Returns: (voice_friendly_english, localized_text, audio_url)
    """
    voice_friendly = voice_engine.prepare_response_for_voice(raw_response)
    localized = voice_engine.translate_from_english(voice_friendly, target_lang)
    audio_name = await synthesize_speech(localized)
    audio_url = build_audio_url(audio_name)
    return voice_friendly, localized, audio_url


def _new_live_session() -> dict:
    return {
        "audio_chunks": [],
        "filename": "live.webm",
        "language": "",
        "thread_id": None,
        "message_hint": "",
        "username": "anonymous",
        "user_role": "client",
        "login_dto": {},
        "created_at": time.time(),
        "updated_at": time.time(),
    }


async def _process_live_audio_session(session: dict) -> dict:
    """
    Process accumulated browser audio chunks and return the voice response payload.
    This reuses the existing orchestrator without altering chatbot behavior.
    """
    from orchestrator_main import ai_orchestrator, history_manager  # local import to avoid circular startup issues

    audio_bytes = b"".join(session.get("audio_chunks", []))
    audio_name = session.get("filename") or "live.webm"
    username = session.get("username", "anonymous")
    user_role = session.get("user_role", "client")
    login_dto = session.get("login_dto", {})
    thread_id = session.get("thread_id")
    language = (session.get("language") or "").strip().lower()
    message_hint = (session.get("message_hint") or "").strip()

    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Audio stream is too large.")

    transcribed_text = await transcribe_audio_bytes(audio_bytes, audio_name)
    user_input = (transcribed_text or message_hint).strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="No valid speech was captured.")

    detected_lang = language if language else voice_engine.detect_language(user_input)
    english_input = voice_engine.translate_to_english(user_input, detected_lang)

    _thread_obj = await asyncio.to_thread(history_manager.get_thread, thread_id) if thread_id else None
    is_existing = bool(thread_id and _thread_obj)
    if not thread_id:
        thread_id = await asyncio.to_thread(history_manager.create_new_thread, username, english_input)

    result = await ai_orchestrator.process_request(
        username,
        user_role,
        english_input,
        thread_id,
        is_existing_thread=is_existing,
        login_dto=login_dto,
    )

    raw_response = result.get("response", "")
    bot_type = result.get("bot_type", "general")
    voice_text, localized_response, audio_url = await clean_and_speak_response(raw_response, detected_lang)
    await asyncio.to_thread(cleanup_old_audio_files)

    return {
        "success": True,
        "detected_language": detected_lang,
        "transcribed_text": transcribed_text or user_input,
        "translated_text": english_input,
        "chatbot_response": voice_text,
        "localized_response": localized_response,
        "audio_url": audio_url,
        "thread_id": thread_id,
        "bot_type": bot_type,
    }


@router.websocket("/voice_stream")
async def voice_stream(websocket: WebSocket):
    """
    Live mic capture endpoint.
    Browser sends MediaRecorder chunks as binary frames and a final {"type":"stop"} message.
    """
    await websocket.accept()
    session_id = uuid.uuid4().hex
    session = _new_live_session()
    _LIVE_SESSIONS[session_id] = session

    try:
        await websocket.send_json({
            "type": "ready",
            "session_id": session_id,
            "message": "Voice stream ready. Send binary audio chunks, then {\"type\":\"stop\"}.",
        })

        while True:
            message = await websocket.receive()
            session["updated_at"] = time.time()

            if message.get("bytes") is not None:
                chunk = message["bytes"]
                if chunk:
                    session["audio_chunks"].append(chunk)
                await websocket.send_json({
                    "type": "chunk_ack",
                    "bytes_received": len(chunk or b""),
                    "total_bytes": sum(len(c) for c in session["audio_chunks"]),
                })
                continue

            text = (message.get("text") or "").strip()
            if not text:
                continue

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON control message."})
                continue

            msg_type = (payload.get("type") or "").lower()

            if msg_type == "start":
                session["filename"] = payload.get("filename") or "live.webm"
                session["language"] = (payload.get("language") or "").strip().lower()
                session["thread_id"] = payload.get("thread_id")
                session["message_hint"] = (payload.get("message_hint") or "").strip()
                session["username"] = payload.get("username") or session["username"]
                session["user_role"] = (payload.get("user_role") or session["user_role"]).lower()
                session["login_dto"] = payload.get("login_dto") or session["login_dto"]
                await websocket.send_json({"type": "started", "session_id": session_id})
                continue

            if msg_type == "stop":
                try:
                    result = await _process_live_audio_session(session)
                    await websocket.send_json({"type": "result", **result})
                finally:
                    _LIVE_SESSIONS.pop(session_id, None)
                await websocket.close()
                return

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type or 'empty'}"})
    except WebSocketDisconnect:
        _LIVE_SESSIONS.pop(session_id, None)
    except Exception as exc:
        _LIVE_SESSIONS.pop(session_id, None)
        logger.error(f"[VoiceStream] WebSocket error: {exc}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": "Live voice stream failed."})
        except Exception:
            pass
