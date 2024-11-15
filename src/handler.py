""" Example handler file. """

import runpod
import requests
import os
import torch
import base64
import tempfile
import traceback
import json
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperForConditionalGeneration,
    pipeline,
)
from runpod.serverless.utils import rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
from rp_schema import INPUT_VALIDATIONS
from urllib.parse import urlparse
import hashlib


def base64_to_tempfile(base64_file: str) -> str:
    """
    Convert base64 file to tempfile.

    Parameters:
    base64_file (str): Base64 file

    Returns:
    str: Path to tempfile
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))

    return temp_file.name


def download_file(url):
    """
    Download file from URL.
    """
    # Parse URL
    parsed_url = urlparse(url)
    file_name = os.path.basename(parsed_url.path)
    print(f"Downloading file: {file_name} from {url}")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=file_name, delete=False) as temp_file:
        try:
            response = requests.get(url)
            response.raise_for_status()
            temp_file.write(response.content)
        except requests.exceptions.RequestException as e:
            os.remove(temp_file.name)
            raise e

    return temp_file.name


def run_whisper_inference(
    model_id,
    audio_path,
    chunk_length,
    batch_size,
    language,
    task,
    return_timestamps,
):
    """Run Whisper model inference on the given audio file."""
    try:
        torch_dtype = torch.float16
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_cache = "/cache/huggingface/hub"
        local_files_only = True

        # Load the model with Flash Attention 2
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            cache_dir=model_cache,
            local_files_only=local_files_only,
            # device=device,
            # use_flash_attention_2=True,  # Enable Flash Attention 2
        )

        tokenizer = WhisperTokenizerFast.from_pretrained(
            model_id, cache_dir=model_cache, local_files_only=local_files_only
        )
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_id, cache_dir=model_cache, local_files_only=local_files_only
        )

        # Initialize pipeline with better memory management
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            # model_kwargs={"use_flash_attention_2": True},
            torch_dtype=torch_dtype,
            device=device,
        )

        # Clear CUDA cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        outputs = pipe(
            audio_path,
            chunk_length_s=chunk_length,
            batch_size=batch_size,
            generate_kwargs={"task": task, "language": language},
            return_timestamps=return_timestamps,
        )

        outputs = json.loads(json.dumps(outputs))
        return outputs

    except torch.cuda.OutOfMemoryError:
        # Handle CUDA OOM specifically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA out of memory. Try reducing batch_size or chunk_length."
        )
    except Exception as e:
        raise RuntimeError(f"Whisper inference failed: {str(e)}")


@rp_debugger.FunctionTimer
def handler(job):
    try:
        job_input = job["input"]
        print(f"Received job: {job_input}")

        # Validate input
        print("Validating input")
        with rp_debugger.LineTimer("validation_step"):
            input_validation = validate(job_input, INPUT_VALIDATIONS)
            if "errors" in input_validation:
                return {"error": input_validation["errors"]}
            job_input = input_validation["validated_input"]

        print("Validated input")
        for key, value in job_input.items():
            print(f"{key}: {value}")

        if not job_input.get("audio", False) and not job_input.get(
            "audio_base64", False
        ):
            return {"error": "Must provide either audio or audio_base64"}

        if job_input.get("audio", False) and job_input.get("audio_base64", False):
            return {"error": "Must provide either audio or audio_base64, not both"}

        print(f"Running job")

        # Handle audio input
        audio_file_path = None
        try:
            if job_input.get("audio", None):
                with rp_debugger.LineTimer("download_step"):
                    audio_file_path = download_file(job_input["audio"])
            elif job_input.get("audio_base64", None):
                audio_file_path = base64_to_tempfile(job_input["audio_base64"])

            print("Got audio input")

            # Run prediction
            with rp_debugger.LineTimer("prediction_step"):
                result = run_whisper_inference(
                    job_input.get("model", "openai/whisper-large-v3-turbo"),
                    audio_file_path,
                    job_input.get("chunk_length", 30),
                    job_input.get("batch_size", 16),
                    job_input.get("language"),
                    job_input.get("task", "transcribe"),
                    job_input.get("return_timestamps", True),
                )

            print(f"Got whisper results: {result}")
            return {"output": result}
        except Exception as e:
            raise e

        finally:
            with rp_debugger.LineTimer("cleanup_step"):
                rp_cleanup.clean(["input_objects"])
                if audio_file_path and os.path.exists(audio_file_path):
                    os.remove(audio_file_path)

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}\nTraceback:\n{''.join(traceback.format_tb(e.__traceback__))}"
        print(error_msg)
        return {"error": error_msg}


runpod.serverless.start({"handler": handler})
