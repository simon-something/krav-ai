"""Gemini video analyzer using OpenRouter API."""

import base64
import json
import os
import re

import httpx

from backend.prompts.gemini_system import get_gemini_prompt

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODEL = "google/gemini-2.5-pro"

_MIME_TYPES = {
    ".webm": "video/webm",
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class GeminiAnalyzer:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def _build_messages(
        self,
        video_path: str,
        strike_type: str,
        reference_image_paths: list[str],
    ) -> list[dict]:
        """Build the multimodal messages list for the OpenRouter API.

        Returns a list with a single user message containing text + video +
        reference image parts.
        """
        prompt_text = get_gemini_prompt(strike_type)

        content_parts: list[dict] = [{"type": "text", "text": prompt_text}]

        # Video part
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        video_ext = os.path.splitext(video_path)[1].lower()
        video_mime = _MIME_TYPES.get(video_ext, "video/webm")
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{video_mime};base64,{video_b64}"},
            }
        )

        # Reference image parts
        for img_path in reference_image_paths:
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            img_ext = os.path.splitext(img_path)[1].lower()
            img_mime = _MIME_TYPES.get(img_ext, "image/jpeg")
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img_mime};base64,{img_b64}"},
                }
            )

        return [{"role": "user", "content": content_parts}]

    def _parse_json_response(self, text: str) -> dict:
        """Extract and parse JSON from model response text.

        Handles cases where the model wraps JSON in markdown code blocks.
        """
        # Try stripping markdown code fences first
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
        return json.loads(text)

    def analyze(
        self,
        video_path: str,
        strike_type: str,
        reference_image_paths: list[str],
    ) -> dict:
        """Analyze a strike video and return structured observations.

        Sends the video and reference images to Gemini 2.5 Pro via OpenRouter.
        Retries once on HTTP or JSON parse failure.

        Args:
            video_path: Path to the video file.
            strike_type: One of "jab", "cross", or "jab_cross".
            reference_image_paths: Paths to reference technique images.

        Returns:
            Parsed observation dict matching the OBSERVATION_SCHEMA structure.

        Raises:
            RuntimeError: If both attempts fail.
        """
        messages = self._build_messages(video_path, strike_type, reference_image_paths)
        payload = {"model": _MODEL, "messages": messages}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None

        for _attempt in range(2):
            try:
                with httpx.Client(timeout=120.0) as client:
                    response = client.post(
                        _OPENROUTER_URL,
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()

                data = response.json()
                text = data["choices"][0]["message"]["content"]
                return self._parse_json_response(text)

            except (httpx.HTTPStatusError, json.JSONDecodeError, KeyError) as exc:
                last_error = exc

        raise RuntimeError(
            f"Gemini analysis failed after 2 attempts: {last_error}"
        ) from last_error
