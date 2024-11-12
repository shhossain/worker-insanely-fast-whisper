INPUT_VALIDATIONS = {
    "model": {
        "type": str,
        "required": False,
        "default": "openai/whisper-large-v3-turbo",
    },
    "audio": {"type": str, "required": False, "default": None},
    "audio_base64": {"type": str, "required": False, "default": None},
    "chunk_length": {"type": int, "required": False, "default": 30},
    "batch_size": {"type": int, "required": False, "default": 16},
    "language": {"type": str, "required": False, "default": None},
    "task": {"type": str, "required": False, "default": "transcribe"},
    "return_timestamps": {"type": str | bool, "required": False, "default": True},
}
