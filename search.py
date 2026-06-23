"""
search.py
---------
AI Pattern Search ╬ô├ç├╢ Bedrock retrieval, S3 presigning, and reranking.
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
#  Boto3 clients ╬ô├ç├╢ module-level so they are reused across requests
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
    to prevent double-encoding. boto3 handles '#' and other special
    characters correctly when generating the presigned URL.
    Returns None on failure.
    """
    if not key:
        return None
    try:
        clean_key = unquote(key).replace("\\", "/")
        return _s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": IMAGE_BUCKET, "Key": clean_key},
            ExpiresIn=expires,
        )
    except Exception as e:
        log.warning("Pre-sign error for %s: %s", key, e)
        return None

# ------------------------------------------------------------------ #
#  Score normalisation
# ------------------------------------------------------------------ #

def _normalize_scores(rugs: list) -> list:
    """
    Scale raw Bedrock similarity/rerank scores so the top result sits
    at ~95% and the bottom at ~70%, giving a stable, readable range
    regardless of query type or expansion strategy.
    The relative ordering is preserved exactly.
    """
    if not rugs:
        return rugs
    high   = rugs[0]["score"]
    low    = rugs[-1]["score"]
    spread = high - low or 1
    for rug in rugs:
        rug["score"] = round(70 + ((rug["score"] - low) / spread) * 25)
    return rugs


# ------------------------------------------------------------------ #
#  Retrieve rugs
# ------------------------------------------------------------------ #

def retrieve_rugs(query, max_results=DEFAULT_MAX_RESULTS, exclude_ids=None, year_filter=None):
    exclude = set(exclude_ids or [])

    try:
        response = _bedrock.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": min(max_results, 100)}
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
            _id_match = re.match(r"^(\d+)", fname)
            rug_id = _id_match.group(1) if _id_match else ""
            if not rug_id:
                continue
            # Apply year filter — URI path must contain /{year}/
            if year_filter and f"/{year_filter}/" not in uri:
                continue
            score = round(hit.get("score", 0) * 100)
            if rug_id not in seen or score > seen[rug_id]["score"]:
                seen[rug_id] = {"score": score, "fname": fname}
        except Exception as e:
            log.debug("Skipping hit: %s", e)

    # True size variants (same design, different dimensions) share the same rug_id
    # (first "-"-segment of the filename) and are already collapsed by the seen dict
    # above. No further dedup needed.
    candidates = {k: v for k, v in seen.items() if k not in exclude}
    log.info("Candidates after dedup: %d", len(candidates))

    if not candidates:
        return []

    rugs = []
    for rug_id, meta in candidates.items():
        try:
            json_key = JSON_PREFIX + unquote(meta["fname"])
            log.debug("Attempting S3 key: %s", json_key)
            obj      = _s3.get_object(Bucket=JSON_BUCKET, Key=json_key)
            data     = json.loads(obj["Body"].read().decode("utf-8"))

            analysis = data.get("analysis", {})
            source   = data.get("source_config", {})

            img_key = source.get("s3_image_key", "") or unquote(meta["fname"])
            img_url = presign(img_key)

            size_match = re.search(r"-(\d+)x(\d+)-", meta["fname"])
            width  = size_match.group(1) if size_match else data.get("width",  "╬ô├ç├╢")
            height = size_match.group(2) if size_match else data.get("height", "╬ô├ç├╢")

            rugs.append({
                "rug_id":           rug_id,
                "_img_key":         img_key,
                "img_url":          img_url,
                "score":            meta["score"],
                "style":            analysis.get("style",            "╬ô├ç├╢"),
                "pattern_type":     analysis.get("pattern_type",     "╬ô├ç├╢"),
                "primary_colors":   analysis.get("primary_colors",   []),
                "secondary_colors": analysis.get("secondary_colors", []),
                "design_elements":  analysis.get("design_elements",  []),
                "tone":             analysis.get("tone",             "╬ô├ç├╢"),
                "complexity":       analysis.get("complexity",       "╬ô├ç├╢"),
                "origin":           analysis.get("origin",           "╬ô├ç├╢"),
                "material":         analysis.get("material",         "╬ô├ç├╢"),
                "width":            width,
                "height":           height,
                "description":      analysis.get("description_raw",  ""),
            })
            log.info("Fetched rug %s ok", rug_id)
        except _s3.exceptions.NoSuchKey:
            # Exact key not found — search by rug_id prefix
            prefix = JSON_PREFIX + rug_id + "-"
            log.debug("Prefix search: bucket=%s prefix=%s", JSON_BUCKET, prefix)
            results = _s3.list_objects_v2(Bucket=JSON_BUCKET, Prefix=prefix, MaxKeys=1)
            contents = results.get("Contents", [])
            if not contents:
                log.warning("No JSON found for rug %s (searched %s%s-*)", rug_id, JSON_BUCKET, prefix)
                continue
            json_key = contents[0]["Key"]
            log.info("Found via prefix search: %s", json_key)
            try:
                obj  = _s3.get_object(Bucket=JSON_BUCKET, Key=json_key)
                data = json.loads(obj["Body"].read().decode("utf-8"))

                analysis = data.get("analysis", {})
                source   = data.get("source_config", {})

                img_key = source.get("s3_image_key", "") or unquote(meta["fname"])
                img_url = presign(img_key)

                size_match = re.search(r"-(\d+)x(\d+)-", meta["fname"])
                width  = size_match.group(1) if size_match else data.get("width",  "?")
                height = size_match.group(2) if size_match else data.get("height", "?")

                rugs.append({
                    "rug_id":           rug_id,
                    "_img_key":         img_key,
                    "img_url":          img_url,
                    "score":            meta["score"],
                    "style":            analysis.get("style",            "?"),
                    "pattern_type":     analysis.get("pattern_type",     "?"),
                    "primary_colors":   analysis.get("primary_colors",   []),
                    "secondary_colors": analysis.get("secondary_colors", []),
                    "design_elements":  analysis.get("design_elements",  []),
                    "tone":             analysis.get("tone",             "?"),
                    "complexity":       analysis.get("complexity",       "?"),
                    "origin":           analysis.get("origin",           "?"),
                    "material":         analysis.get("material",         "?"),
                    "width":            width,
                    "height":           height,
                    "description":      analysis.get("description_raw",  ""),
                })
                log.info("Fetched rug %s ok (prefix fallback)", rug_id)
            except Exception as e:
                log.error("Failed to fetch JSON for %s via prefix: %s", rug_id, e, exc_info=True)
        except Exception as e:
            log.error("Failed to fetch JSON for %s: %s", rug_id, e, exc_info=True)

    # --- Step 5b: Deduplicate by source image key ---
    # Size variants of the same design share the same source image file.
    # Keep the highest-scoring entry per unique image key.
    seen_img: dict = {}
    for rug in rugs:
        key = rug.get("_img_key") or rug["rug_id"]
        if key not in seen_img or rug["score"] > seen_img[key]["score"]:
            seen_img[key] = rug
    rugs = list(seen_img.values())
    # Strip internal key before returning
    for rug in rugs:
        rug.pop("_img_key", None)

    # --- Step 6: Sort by score ---
    rugs.sort(key=lambda r: r["score"], reverse=True)
    rugs = rugs[:max_results]
    rugs = _normalize_scores(rugs)
    log.info("retrieve_rugs returning %d results (scores normalized)", len(rugs))
    return rugs

# ------------------------------------------------------------------ #
#  Multi-query fan-out
# ------------------------------------------------------------------ #

def retrieve_rugs_multi(queries, max_results=DEFAULT_MAX_RESULTS, exclude_ids=None, year_filter=None):
    """
    Fan out multiple query strings to Bedrock in parallel, merge by rug_id
    keeping the highest score, then run the normal S3-fetch + dedup pipeline.

    Used in broad mode so each expanded synonym gets its own vector search
    rather than one combined query that dilutes recall.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    exclude = set(exclude_ids or [])
    per_query = 100  # always request Bedrock's max per term to maximise unique pool

    # Fire one Bedrock retrieve per query term in parallel
    merged_seen: dict = {}  # rug_id → {score, fname}

    def _retrieve_one(q):
        try:
            resp = _bedrock.retrieve(
                knowledgeBaseId=KNOWLEDGE_BASE_ID,
                retrievalQuery={"text": q},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": per_query}
                },
            )
            return q, resp.get("retrievalResults", [])
        except Exception as e:
            log.error("Bedrock retrieve failed for %r: %s", q, e)
            return q, []

    with ThreadPoolExecutor(max_workers=min(len(queries), 8)) as pool:
        futures = [pool.submit(_retrieve_one, q) for q in queries]
        for fut in as_completed(futures):
            q, results = fut.result()
            log.info("Bedrock returned %d hits for %r", len(results), q)
            for hit in results:
                try:
                    uri    = hit.get("location", {}).get("s3Location", {}).get("uri", "")
                    fname  = uri.split("/")[-1]
                    rug_id = fname.split("-")[0]
                    if not rug_id:
                        continue
                    if year_filter and f"/{year_filter}/" not in uri:
                        continue
                    score = round(hit.get("score", 0) * 100)
                    if rug_id not in merged_seen or score > merged_seen[rug_id]["score"]:
                        merged_seen[rug_id] = {"score": score, "fname": fname}
                except Exception as e:
                    log.debug("Skipping hit: %s", e)

    candidates = {k: v for k, v in merged_seen.items() if k not in exclude}
    log.info("Multi-query candidates after merge+exclude: %d", len(candidates))

    if not candidates:
        return []

    # Reuse the S3-fetch + image-dedup + normalize logic from retrieve_rugs
    # by constructing a temporary seen dict and delegating to a shared helper.
    # Simplest approach: call retrieve_rugs with a combined query string but
    # override the Bedrock step by patching candidates directly.  Instead we
    # just duplicate the fetch loop here — it's self-contained.
    rugs = []
    for rug_id, meta in candidates.items():
        try:
            json_key = JSON_PREFIX + unquote(meta["fname"])
            obj      = _s3.get_object(Bucket=JSON_BUCKET, Key=json_key)
            data     = json.loads(obj["Body"].read().decode("utf-8"))
        except _s3.exceptions.NoSuchKey:
            prefix   = JSON_PREFIX + rug_id + "-"
            results  = _s3.list_objects_v2(Bucket=JSON_BUCKET, Prefix=prefix, MaxKeys=1)
            contents = results.get("Contents", [])
            if not contents:
                log.warning("No JSON for rug %s", rug_id)
                continue
            try:
                obj  = _s3.get_object(Bucket=JSON_BUCKET, Key=contents[0]["Key"])
                data = json.loads(obj["Body"].read().decode("utf-8"))
                meta["fname"] = contents[0]["Key"].split("/")[-1]
            except Exception as e:
                log.error("Prefix fallback failed for %s: %s", rug_id, e)
                continue
        except Exception as e:
            log.error("S3 fetch failed for %s: %s", rug_id, e)
            continue

        analysis   = data.get("analysis", {})
        source     = data.get("source_config", {})
        img_key    = source.get("s3_image_key", "") or unquote(meta["fname"])
        img_url    = presign(img_key)
        size_match = re.search(r"-(\d+)x(\d+)-", meta["fname"])
        width      = size_match.group(1) if size_match else data.get("width",  "?")
        height     = size_match.group(2) if size_match else data.get("height", "?")

        rugs.append({
            "rug_id":           rug_id,
            "_img_key":         img_key,
            "img_url":          img_url,
            "score":            meta["score"],
            "style":            analysis.get("style",            "?"),
            "pattern_type":     analysis.get("pattern_type",     "?"),
            "primary_colors":   analysis.get("primary_colors",   []),
            "secondary_colors": analysis.get("secondary_colors", []),
            "design_elements":  analysis.get("design_elements",  []),
            "tone":             analysis.get("tone",             "?"),
            "complexity":       analysis.get("complexity",       "?"),
            "origin":           analysis.get("origin",           "?"),
            "material":         analysis.get("material",         "?"),
            "width":            width,
            "height":           height,
            "description":      analysis.get("description_raw",  ""),
        })
        log.info("Fetched rug %s ok (multi)", rug_id)

    # Image-key dedup — same image = same design
    seen_img: dict = {}
    for rug in rugs:
        key = rug.get("_img_key") or rug["rug_id"]
        if key not in seen_img or rug["score"] > seen_img[key]["score"]:
            seen_img[key] = rug
    rugs = list(seen_img.values())
    for rug in rugs:
        rug.pop("_img_key", None)

    rugs.sort(key=lambda r: r["score"], reverse=True)
    rugs = rugs[:max_results]
    rugs = _normalize_scores(rugs)
    log.info("retrieve_rugs_multi returning %d results", len(rugs))
    return rugs


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
        reranked = _normalize_scores(reranked)
        log.info("rerank_rugs: %d results reranked (scores normalized)", len(reranked))
        return reranked
    except Exception as e:
        log.warning("Reranking failed, using original order: %s", e)
        return rugs
