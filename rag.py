"""
rag.py
------
AI Pattern Search — RAG and query expansion.

Two functions:
  - expand_query()   : uses Claude to enrich a short user query with
                       synonyms, design terms, and taxonomy keywords
                       before it hits the Bedrock knowledge base.

  - rag_summarise()  : calls Bedrock RetrieveAndGenerate to produce
                       a professional natural-language summary of
                       what was found. Optionally incorporates
                       conversation history for context continuity.
"""

import json
import logging
import urllib.request
import urllib.error

import boto3

from config import (
    AWS_REGION,
    KNOWLEDGE_BASE_ID,
    BEDROCK_MODEL_ARN,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    ANTHROPIC_URL,
)
from keywords import taxonomy_str

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Boto3 client — module-level so it's reused across requests
# ------------------------------------------------------------------ #

_bedrock = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

# ------------------------------------------------------------------ #
#  RAG system prompt
# ------------------------------------------------------------------ #

_RAG_SYSTEM = (
    "You are an expert rug analyst for a high-end rug retailer. "
    "Help sales representatives and design teams find the right pattern "
    "for each project. Use professional, sales-oriented language referencing "
    "specific patterns, colour palettes, and design characteristics. "
    "Be concise — 2-3 sentences maximum."
)

# ------------------------------------------------------------------ #
#  Query expansion
# ------------------------------------------------------------------ #

def expand_query(query: str) -> str:
    """
    Use Claude to enrich the user's query with synonyms, design vocabulary,
    and terms from the official keyword taxonomy before semantic search.

    Falls back to the original query if the API call fails.
    """
    taxonomy = taxonomy_str()
    payload = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 200,
        "system": (
            "You are a rug industry search expert. Expand the user's search "
            "query into richer terms that would appear in professional rug "
            "descriptions. Include synonyms, related design elements, pattern "
            "names, and colour variations. Where relevant, reference terms from "
            f"this official product taxonomy: {taxonomy}. "
            "Return only the expanded query as a single paragraph — "
            "no explanation, no preamble."
        ),
        "messages": [
            {"role": "user", "content": f"Expand this rug search query: {query}"}
        ],
    }).encode()

    req = urllib.request.Request(
        ANTHROPIC_URL,
        data=payload,
        headers={
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type":      "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data     = json.loads(resp.read().decode())
            expanded = data["content"][0]["text"]
            log.info("Query expanded: '%s' -> '%s...'", query[:50], expanded[:80])
            return expanded
    except Exception as e:
        log.warning("Query expansion failed, using original: %s", e)
        return query


# ------------------------------------------------------------------ #
#  RAG summary
# ------------------------------------------------------------------ #

def rag_summarise(query: str, history = None) -> str:
    """
    Call Bedrock RetrieveAndGenerate to produce a natural-language summary
    of search results.

    If history is provided (conversation mode), the last 3 prior queries
    are prepended as context so the summary acknowledges what was
    previously shown.

    Returns a plain-text summary string, or a fallback message on error.
    """
    # Build context prefix for conversation mode
    context = ""
    if history:
        prior = history[-3:]
        context = "Previous searches in this session:\n"
        context += "".join(f'- "{h["query"]}"\n' for h in prior)
        context += "\nCurrent search: "

    input_text = context + query

    try:
        response = _bedrock.retrieve_and_generate(
            input={"text": input_text},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn":        BEDROCK_MODEL_ARN,
                    "generationConfiguration": {
                        "promptTemplate": {
                            "textPromptTemplate": (
                                _RAG_SYSTEM
                                + "\n\n$search_results$\n\n$output_format_instructions$"
                            )
                        }
                    },
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {"numberOfResults": 10}
                    },
                },
            },
        )
        summary = response["output"]["text"]
        log.debug("RAG summary produced (%d chars)", len(summary))
        return summary

    except Exception as e:
        log.error("RAG summarise error: %s", e)
        return "Results retrieved — summary unavailable."
