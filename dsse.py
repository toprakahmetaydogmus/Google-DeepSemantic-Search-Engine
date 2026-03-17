import os
import re
import json
import math
import time
import mimetypes
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
import google.auth
from google.auth.transport.requests import Request
from google.cloud import storage

# ============================================================
# GEMINI MULTIMODAL INDEXER
# - Input:  GCS bucket with images + videos
# - Output: GCS JSON index with Gemini-generated semantic summaries
#           + Gemini embeddings
#
# Designed for:
#   - Google Cloud Run Jobs
#   - GCE / GKE / Vertex Workbench
#   - Any Google Cloud server with ADC enabled
#
# ------------------------------------------------------------
# REQUIRED ENV VARS
# ------------------------------------------------------------
# GOOGLE_CLOUD_PROJECT=my-project-id
# GOOGLE_CLOUD_LOCATION=global
# INPUT_BUCKET=my-media-bucket
# INPUT_PREFIX=media/
# OUTPUT_BUCKET=my-output-bucket
#
# OPTIONAL ENV VARS
# OUTPUT_BLOB=index/index.json
# GEMINI_MODEL=gemini-2.5-flash
# EMBEDDING_MODEL=gemini-embedding-001
# MAX_ITEMS=0
# TEXT_EMBED_DIM=1536
# QUERY_SMOKE_TEST=dancing laughing bird
# REQUEST_TIMEOUT=300
# LOG_LEVEL=INFO
#
# ------------------------------------------------------------
# IAM NEEDED
# ------------------------------------------------------------
# - roles/storage.objectViewer on input bucket
# - roles/storage.objectCreator or objectAdmin on output bucket
# - Vertex AI User (or equivalent access to Gemini on Vertex AI)
#
# ------------------------------------------------------------
# pip install
# ------------------------------------------------------------
# pip install google-cloud-storage google-auth requests
# ============================================================

# ----------------------------
# CONFIG
# ----------------------------
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global").strip()
INPUT_BUCKET = os.environ.get("INPUT_BUCKET", "").strip()
INPUT_PREFIX = os.environ.get("INPUT_PREFIX", "media/").strip()
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "").strip()
OUTPUT_BLOB = os.environ.get("OUTPUT_BLOB", "index/index.json").strip()

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001").strip()
MAX_ITEMS = int(os.environ.get("MAX_ITEMS", "0"))
TEXT_EMBED_DIM = int(os.environ.get("TEXT_EMBED_DIM", "1536"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "300"))
QUERY_SMOKE_TEST = os.environ.get("QUERY_SMOKE_TEST", "").strip()

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("gemini-multimodal-indexer")


# ----------------------------
# VALIDATION
# ----------------------------
def require_env() -> None:
    missing = []
    for key, value in {
        "GOOGLE_CLOUD_PROJECT": PROJECT_ID,
        "INPUT_BUCKET": INPUT_BUCKET,
        "OUTPUT_BUCKET": OUTPUT_BUCKET,
    }.items():
        if not value:
            missing.append(key)

    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


# ----------------------------
# AUTH
# ----------------------------
def get_access_token() -> str:
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return credentials.token


# ----------------------------
# GCS HELPERS
# ----------------------------
def parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    path = gcs_uri[5:]
    bucket, blob = path.split("/", 1)
    return bucket, blob


def list_gcs_media(bucket_name: str, prefix: str) -> List[Dict[str, Any]]:
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)

    items: List[Dict[str, Any]] = []
    for blob in blobs:
        name = blob.name
        lower = name.lower()

        ext = os.path.splitext(lower)[1]
        media_type = None
        if ext in VIDEO_EXTS:
            media_type = "video"
        elif ext in IMAGE_EXTS:
            media_type = "image"
        else:
            continue

        mime_type = blob.content_type or mimetypes.guess_type(name)[0] or (
            "video/mp4" if media_type == "video" else "image/jpeg"
        )

        items.append(
            {
                "gcs_uri": f"gs://{bucket_name}/{name}",
                "blob_name": name,
                "media_type": media_type,
                "mime_type": mime_type,
                "size_bytes": blob.size,
                "updated": blob.updated.isoformat() if blob.updated else None,
            }
        )

    items.sort(key=lambda x: x["blob_name"])
    if MAX_ITEMS > 0:
        items = items[:MAX_ITEMS]
    return items


def upload_json_to_gcs(bucket_name: str, blob_name: str, payload: Dict[str, Any]) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(payload, ensure_ascii=False, indent=2),
        content_type="application/json",
    )


# ----------------------------
# HTTP HELPERS
# ----------------------------
def vertex_generate_content(
    model: str,
    parts: List[Dict[str, Any]],
    response_mime_type: str = "application/json",
    temperature: float = 0.1,
    max_output_tokens: int = 2048,
) -> Dict[str, Any]:
    token = get_access_token()
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{model}:generateContent"
    )

    body = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": response_mime_type,
        },
    }

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json=body,
        timeout=REQUEST_TIMEOUT,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Gemini generateContent failed: {resp.status_code} {resp.text[:2000]}")
    return resp.json()


def vertex_text_embedding(text: str, dimension: int) -> List[float]:
    token = get_access_token()
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{EMBEDDING_MODEL}:predict"
    )

    body = {
        "instances": [{"content": text[:30000]}],
        "parameters": {
            "autoTruncate": True,
            "outputDimensionality": dimension,
        },
    }

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json=body,
        timeout=REQUEST_TIMEOUT,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Gemini embedding failed: {resp.status_code} {resp.text[:2000]}")

    data = resp.json()
    predictions = data.get("predictions", [])
    if not predictions:
        raise RuntimeError(f"No embedding returned: {json.dumps(data)[:1200]}")

    pred = predictions[0]

    # Be defensive with response shape
    if isinstance(pred, dict):
        if "embeddings" in pred and isinstance(pred["embeddings"], dict):
            vals = pred["embeddings"].get("values")
            if isinstance(vals, list):
                return vals

        if "embedding" in pred and isinstance(pred["embedding"], dict):
            vals = pred["embedding"].get("values")
            if isinstance(vals, list):
                return vals

        if "values" in pred and isinstance(pred["values"], list):
            return pred["values"]

    raise RuntimeError(f"Unexpected embedding response: {json.dumps(data)[:1200]}")


# ----------------------------
# JSON EXTRACTION
# ----------------------------
def extract_text_from_gemini_response(resp: Dict[str, Any]) -> str:
    candidates = resp.get("candidates", [])
    if not candidates:
        return ""

    parts = candidates[0].get("content", {}).get("parts", [])
    texts = []
    for part in parts:
        if "text" in part and isinstance(part["text"], str):
            texts.append(part["text"])
    return "\n".join(texts).strip()


def extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse JSON from model output: {text[:1000]}")

    return json.loads(match.group(0))


# ----------------------------
# ANALYSIS
# ----------------------------
ANALYSIS_SCHEMA_EXAMPLE = {
    "media_type": "image_or_video",
    "short_title": "very short semantic title",
    "summary": "compact but meaningful description of what the content represents",
    "objects": ["object1", "object2"],
    "actions": ["action1", "action2"],
    "scene": ["scene_keyword1", "scene_keyword2"],
    "mood": ["mood1"],
    "speech_or_text": "visible text or speech if any, else empty string",
    "search_keywords": ["keyword1", "keyword2", "keyword3"],
    "safety_notes": ["optional non-sensitive notes if relevant"],
}

SYSTEM_STYLE_PROMPT = """
You are building a semantic search index.
Analyze the supplied media and return STRICT JSON only.
Do not wrap JSON in markdown.
Do not explain.
Be factual.
Do not invent uncertain details.
If uncertain, use broad but honest descriptions.
""".strip()


def build_media_prompt(media_type: str) -> str:
    if media_type == "video":
        modality_instructions = """
The file is a VIDEO.
Focus on:
- main objects
- visible actions and motion
- setting / scene
- visible text
- spoken words if obvious from the clip
- overall meaning that would help semantic search
"""
    else:
        modality_instructions = """
The file is an IMAGE.
Focus on:
- main objects
- visible actions or pose
- setting / scene
- visible text
- overall meaning that would help semantic search
"""

    return f"""
{SYSTEM_STYLE_PROMPT}

{modality_instructions}

Return JSON in this exact top-level structure:
{json.dumps(ANALYSIS_SCHEMA_EXAMPLE, ensure_ascii=False)}

Rules:
- summary must be concise but useful for retrieval
- objects/actions/scene/mood/search_keywords must be arrays of short strings
- speech_or_text must be a plain string
- short_title should be 3 to 8 words
- media_type must be "{media_type}"
- search_keywords should be retrieval-oriented, not decorative
""".strip()


def analyze_media_with_gemini(gcs_uri: str, mime_type: str, media_type: str) -> Dict[str, Any]:
    prompt = build_media_prompt(media_type)

    parts = [
        {"text": prompt},
        {
            "fileData": {
                "mimeType": mime_type,
                "fileUri": gcs_uri,
            }
        },
    ]

    resp = vertex_generate_content(
        model=GEMINI_MODEL,
        parts=parts,
        response_mime_type="application/json",
        temperature=0.1,
        max_output_tokens=2048,
    )

    text = extract_text_from_gemini_response(resp)
    data = extract_json_block(text)

    # Normalize output a bit
    out = {
        "media_type": data.get("media_type", media_type),
        "short_title": data.get("short_title", ""),
        "summary": data.get("summary", ""),
        "objects": data.get("objects", []) or [],
        "actions": data.get("actions", []) or [],
        "scene": data.get("scene", []) or [],
        "mood": data.get("mood", []) or [],
        "speech_or_text": data.get("speech_or_text", "") or "",
        "search_keywords": data.get("search_keywords", []) or [],
        "safety_notes": data.get("safety_notes", []) or [],
        "raw_model_text": text,
    }

    for key in ["objects", "actions", "scene", "mood", "search_keywords", "safety_notes"]:
        if not isinstance(out[key], list):
            out[key] = [str(out[key])]

    for key in ["media_type", "short_title", "summary", "speech_or_text", "raw_model_text"]:
        if not isinstance(out[key], str):
            out[key] = str(out[key])

    return out


def build_search_text(item: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    lines = [
        f"uri: {item['gcs_uri']}",
        f"type: {item['media_type']}",
        f"title: {analysis.get('short_title', '')}",
        f"summary: {analysis.get('summary', '')}",
        f"objects: {', '.join(analysis.get('objects', []))}",
        f"actions: {', '.join(analysis.get('actions', []))}",
        f"scene: {', '.join(analysis.get('scene', []))}",
        f"mood: {', '.join(analysis.get('mood', []))}",
        f"speech_or_text: {analysis.get('speech_or_text', '')}",
        f"search_keywords: {', '.join(analysis.get('search_keywords', []))}",
    ]
    return "\n".join(lines).strip()


# ----------------------------
# VECTOR HELPERS
# ----------------------------
def l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


# ----------------------------
# PIPELINE
# ----------------------------
def index_one(item: Dict[str, Any]) -> Dict[str, Any]:
    log.info("Analyzing %s", item["gcs_uri"])

    analysis = analyze_media_with_gemini(
        gcs_uri=item["gcs_uri"],
        mime_type=item["mime_type"],
        media_type=item["media_type"],
    )

    search_text = build_search_text(item, analysis)
    embedding = vertex_text_embedding(search_text, TEXT_EMBED_DIM)
    embedding = l2_normalize(embedding)

    return {
        "gcs_uri": item["gcs_uri"],
        "blob_name": item["blob_name"],
        "media_type": item["media_type"],
        "mime_type": item["mime_type"],
        "size_bytes": item["size_bytes"],
        "updated": item["updated"],
        "analysis": {
            "short_title": analysis["short_title"],
            "summary": analysis["summary"],
            "objects": analysis["objects"],
            "actions": analysis["actions"],
            "scene": analysis["scene"],
            "mood": analysis["mood"],
            "speech_or_text": analysis["speech_or_text"],
            "search_keywords": analysis["search_keywords"],
            "safety_notes": analysis["safety_notes"],
        },
        "search_text": search_text,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": len(embedding),
        "embedding": embedding,
        "indexed_at_unix": int(time.time()),
    }


def build_index(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    indexed_items: List[Dict[str, Any]] = []
    failed_items: List[Dict[str, Any]] = []

    for idx, item in enumerate(items, start=1):
        log.info("(%d/%d) %s", idx, len(items), item["gcs_uri"])
        try:
            indexed = index_one(item)
            indexed_items.append(indexed)
            log.info("Indexed OK: %s", item["gcs_uri"])
        except Exception as e:
            log.exception("Failed: %s", item["gcs_uri"])
            failed_items.append(
                {
                    "gcs_uri": item["gcs_uri"],
                    "error": str(e),
                }
            )

    return {
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "input_bucket": INPUT_BUCKET,
        "input_prefix": INPUT_PREFIX,
        "output_bucket": OUTPUT_BUCKET,
        "output_blob": OUTPUT_BLOB,
        "gemini_model": GEMINI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": TEXT_EMBED_DIM,
        "item_count": len(indexed_items),
        "failed_count": len(failed_items),
        "items": indexed_items,
        "failed": failed_items,
        "created_at_unix": int(time.time()),
        "notes": {
            "querying": "Embed the user query with the same embedding model and rank by cosine similarity against item.embedding.",
            "media": "This index supports both images and videos in one unified semantic retrieval space.",
            "generation": "No synthetic content is generated for results; Gemini is only used to analyze existing content and build semantic metadata.",
        },
    }


# ----------------------------
# OPTIONAL SEARCH TEST
# ----------------------------
def run_smoke_test(index_data: Dict[str, Any], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not query:
        return []

    log.info("Running smoke test query: %s", query)
    qvec = vertex_text_embedding(query, TEXT_EMBED_DIM)
    qvec = l2_normalize(qvec)

    scored: List[Dict[str, Any]] = []
    for item in index_data.get("items", []):
        score = cosine_similarity(qvec, item.get("embedding", []))
        scored.append(
            {
                "gcs_uri": item["gcs_uri"],
                "media_type": item["media_type"],
                "score": round(float(score), 6),
                "short_title": item["analysis"].get("short_title", ""),
                "summary": item["analysis"].get("summary", ""),
                "keywords": item["analysis"].get("search_keywords", [])[:10],
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ----------------------------
# MAIN
# ----------------------------
def main() -> None:
    require_env()

    log.info("Listing media in gs://%s/%s", INPUT_BUCKET, INPUT_PREFIX)
    items = list_gcs_media(INPUT_BUCKET, INPUT_PREFIX)

    if not items:
        raise RuntimeError("No supported image/video files found in input bucket/prefix.")

    log.info("Found %d media file(s)", len(items))
    index_data = build_index(items)

    if QUERY_SMOKE_TEST:
        try:
            index_data["smoke_test"] = {
                "query": QUERY_SMOKE_TEST,
                "results": run_smoke_test(index_data, QUERY_SMOKE_TEST, top_k=5),
            }
        except Exception as e:
            index_data["smoke_test_error"] = str(e)

    upload_json_to_gcs(OUTPUT_BUCKET, OUTPUT_BLOB, index_data)
    log.info("Index written to gs://%s/%s", OUTPUT_BUCKET, OUTPUT_BLOB)


if __name__ == "__main__":
    main()
