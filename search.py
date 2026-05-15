"""
search.py
---------
AI Pattern Search — Bedrock retrieval, S3 presigning, and reranking.
Three responsibilities:
  - presign()       : generate a time-limited S3 URL for a rug image
  - retrieve_rugs() : query Bedrock knowledge base, fetch full JSON
                      from S3 for each result, return structured list
  - rerank_rugs()   : reorder results using Bedrock Cohere reranker
"""

import json
import logging
import re
from urllib.parse import unquote

import boto3

from config import (
    AWS_REGION,
    KNOWLEDGE_BASE_ID,
    IMAGE_BUCKET,
    JSON_BUCKET,
    JSON_PREFIX,
    RERANK_MODEL_ARN,
    PRESIGN_EXPIRY_SECS,
    RERANK_FETCH_MULT,
    DEFAULT_MAX_RESULTS,
)

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Boto3 clients — module-level so they are reused across requests
# ------------------------------------------------------------------ #

_bedrock = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
_s3      = boto3.client("s3",                    region_name=AWS_REGION)

# ------------------------------------------------------------------ #
#  S3 pre-signed URL
# ------------------------------------------------------------------ #

def presign(key, expires=PRESIGN_EXPIRY_SECS):
    """
    Generate a pre-signed S3 URL for an image key.
    Decodes any existing URL encoding on the key before signing
    to prevent double-encoding of special characters (e.g. #).
    Returns None on failure.
    """
    if not key:
        return None
    try:
        clean_key = unquote(key)
        return _s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": IMAGE_BUCKET, "Key": clean_key},
            ExpiresIn=expires,
        )
    except Exception as e:
        log.warning("Pre-sign error for %s: %s", key, e)
        return None

# ------------------------------------------------------------------ #
#  Retrieve rugs
# ------------------------------------------------------------------ #

def retrieve_rugs(query, max_results=DEFAULT_MAX_RESULTS, exclude_ids=None):
    exclude = set(exclude_ids or [])

    try:
        response = _bedrock.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": max_results}
            },
        )
        results = response.get("retrievalResults", [])
        log.info("Bedrock returned %d hits", len(results))
    except Exception as e:
        log.error("Bedrock retrieve failed: %s", e)
        return []

    seen = {}
    for hit in results:
        try:
            uri    = hit.get("location", {}).get("s3Location", {}).get("uri", "")
            fname  = uri.split("/")[-1]
            rug_id = fname.split("-")[0]
            if not rug_id:
                continue
            score = round(hit.get("score", 0) * 100)
            if rug_id not in seen or score > seen[rug_id]["score"]:
                seen[rug_id] = {"score": score, "fname": fname}
        except Exception as e:
            log.debug("Skipping hit: %s", e)

    candidates = {k: v for k, v in seen.items() if k not in exclude}
    log.info("Candidates: %d", len(candidates))

    if not candidates:
        return []

    rugs = []
    for rug_id, meta in candidates.items():
        try:
            json_key = JSON_PREFIX + unquote(meta["fname"])
            obj      = _s3.get_object(Bucket=JSON_BUCKET, Key=json_key)
            data     = json.loads(obj["Body"].read().decode("utf-8"))

            analysis = data.get("analysis", {})
            source   = data.get("source_config", {})

            img_key = source.get("s3_image_key", "") or unquote(meta["fname"])
            img_url = presign(img_key)

            size_match = re.search(r"-(\d+)x(\d+)-", meta["fname"])
            width  = size_match.group(1) if size_match else data.get("width",  "—")
            height = size_match.group(2) if size_match else data.get("height", "—")

            rugs.append({
                "rug_id":           rug_id,
                "img_url":          img_url,
                "score":            meta["score"],
                "style":            analysis.get("style",            "—"),
                "pattern_type":     analysis.get("pattern_type",     "—"),
                "primary_colors":   analysis.get("primary_colors",   []),
                "secondary_colors": analysis.get("secondary_colors", []),
                "design_elements":  analysis.get("design_elements",  []),
                "tone":             analysis.get("tone",             "—"),
                "complexity":       analysis.get("complexity",       "—"),
                "origin":           analysis.get("origin",           "—"),
                "material":         analysis.get("material",         "—"),
                "width":            width,
                "height":           height,
                "description":      analysis.get("description_raw",  ""),
            })
            log.info("Fetched rug %s ok", rug_id)
        except Exception as e:
            log.error("Failed to fetch JSON for %s: %s", rug_id, e, exc_info=True)

    # --- Step 6: Sort by score ---
    rugs.sort(key=lambda r: r["score"], reverse=True)
    log.info("retrieve_rugs returning %d results", len(rugs))
    return rugs[:max_results]

# ------------------------------------------------------------------ #
#  Reranking
# ------------------------------------------------------------------ #

def rerank_rugs(query, rugs):
    if not rugs:
        return rugs
    try:
        sources = [
            {
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {
                        "text": (
                            f"Style: {r['style']}. "
                            f"Pattern: {r['pattern_type']}. "
                            f"Colors: {', '.join(r['primary_colors'])}. "
                            f"Tone: {r['tone']}. "
                            f"{r['description'][:300]}"
                        )
                    },
                },
            }
            for r in rugs
        ]
        response = _bedrock.rerank(
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {"modelArn": RERANK_MODEL_ARN},
                    "numberOfResults": len(rugs),
                },
            },
            sources=sources,
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
        )
        reranked = []
        for item in response["results"]:
            rug = dict(rugs[item["index"]])
            rug["score"] = round(item["relevanceScore"] * 100)
            reranked.append(rug)
        log.info("rerank_rugs: %d results reranked", len(reranked))
        return reranked
    except Exception as e:
        log.warning("Reranking failed, using original order: %s", e)
        return rugs
