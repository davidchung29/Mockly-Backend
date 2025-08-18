"""
- David Chung
- Returns AI-powered STAR analysis using Mistral-7B-Instruct-v0.2
"""

import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration constants
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
# Allow overriding the model via STAR_MODEL; default to a commonly-available instruct model
MISTRAL_MODEL = os.getenv("STAR_MODEL", "mistralai/mistral-medium-3.1")

# Default response for error cases
DEFAULT_STAR_RESPONSE = {
    "situation": [],
    "task": [],
    "action": [],
    "result": [],
    "score": 0
}

async def analyze_star_structure(transcript: str):
    """
    Analyze interview response using STAR format with openai/gpt-5-nano
    """
    if not OPENROUTER_KEY:
        print("Warning: OPENROUTER_API_KEY not found")
        return DEFAULT_STAR_RESPONSE

    prompt = f"""
Given the following candidate's interview response, break it down into STAR format (Situation, Task, Action, Result).

For each category, extract the relevant sentences if they are clear. If a category is not descriptively present, use an empty array. 

Then, assign an objective score from 0 to 75 based on clarity, completeness, and impact of the response, regardless of STAR.

Respond with *only* valid JSON in this exact format:
{{
  "situation": ["Exact sentence/phrase from response describing context or background"],
  "task": ["Exact sentence/phrase from response describing what needed to be accomplished"],
  "action": ["Exact sentence/phrase from response describing what you did"],
  "result": ["Exact sentence/phrase from response describing the outcomes and impact"],
  "score": [Score from 0-75, inclusive]
}}

Interview Response:
\"\"\"
{transcript}
\"\"\"
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        # Ask the provider to return valid JSON only (for models that support it)
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 500,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(OPENROUTER_API, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()

            # Surface API-level errors explicitly if present
            if isinstance(data, dict) and data.get("error"):
                raise ValueError(f"Provider error: {data.get('error')}")

            # Extract content or parsed JSON robustly from the provider response
            content = None
            message = None
            try:
                message = data["choices"][0]["message"]
            except Exception:
                message = None

            # If using JSON mode, some providers include a parsed object
            if isinstance(message, dict) and isinstance(message.get("parsed"), dict):
                result = message["parsed"]

                def ensure_list(value):
                    if isinstance(value, list):
                        return [str(v) for v in value]
                    if value is None:
                        return []
                    if isinstance(value, str):
                        return [value]
                    return [str(value)]

                raw_score = result.get("score", 0)
                try:
                    score_value = int(raw_score) if not isinstance(raw_score, list) else int(raw_score[0])
                except Exception:
                    try:
                        score_value = int(float(raw_score))
                    except Exception:
                        score_value = 0

                return {
                    "situation": ensure_list(result.get("situation", [])),
                    "task": ensure_list(result.get("task", [])),
                    "action": ensure_list(result.get("action", [])),
                    "result": ensure_list(result.get("result", [])),
                    "score": score_value,
                }

            # Fallback: look for string content
            if isinstance(message, dict):
                content = message.get("content")
            if content is None:
                try:
                    content = data["choices"][0]["message"]["content"]
                except Exception:
                    content = None

            # Some providers may return a list of parts; normalize to string
            if isinstance(content, list):
                content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)

            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty or missing content from model response")

            # Try strict JSON parse first
            try:
                result = json.loads(content)
            except Exception:
                # Best-effort: extract the first JSON object from the text
                import re
                match = re.search(r"\{[\s\S]*\}", content)
                if not match:
                    raise ValueError("Model returned non-JSON content")
                result = json.loads(match.group(0))

            return {
                "situation": result.get("situation", []),
                "task": result.get("task", []),
                "action": result.get("action", []),
                "result": result.get("result", []),
                "score": result.get("score", 0)
            }
            
        except Exception as e:
            # Log truncated provider content to aid debugging without flooding logs
            try:
                truncated = content[:300].replace("\n", " ") if isinstance(content, str) else str(content)
                print("Error in STAR analysis:", e, "| model content (truncated):", truncated)
            except Exception:
                print("Error in STAR analysis:", e)
            return DEFAULT_STAR_RESPONSE