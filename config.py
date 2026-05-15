"""
config.py
---------
Central configuration for AI Pattern Search.
All constants, paths, and environment variables live here.
Import this module in all other modules — never hardcode values elsewhere.
"""

import os
from pathlib import Path

# ------------------------------------------------------------------ #
#  AWS
# ------------------------------------------------------------------ #
AWS_REGION        = "us-east-1"
KNOWLEDGE_BASE_ID = "DY4HCOVIDY"
IMAGE_BUCKET      = "variety-bucket-514316422605-us-east-1-an"
JSON_BUCKET       = "result-buckett"
JSON_PREFIX       = "aws/bedrock/knowledge_bases/ASC-VAR/"
BEDROCK_MODEL_ARN = "arn:aws:bedrock:us-east-1:514316422605:inference-profile/us.anthropic.claude-sonnet-4-6"
RERANK_MODEL_ARN  = "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"

# Credentials live here rather than ~/.aws so the service account
# doesn't need a home directory AWS config
AWS_CREDS_DIR     = Path("/apps/python/patternai/aws")
AWS_CREDS_FILE    = AWS_CREDS_DIR / "credentials"
AWS_CONFIG_FILE   = AWS_CREDS_DIR / "config"

# ------------------------------------------------------------------ #
#  Anthropic
# ------------------------------------------------------------------ #
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = "claude-sonnet-4-6"
ANTHROPIC_URL     = "https://api.anthropic.com/v1/messages"


# ------------------------------------------------------------------ #
#  Application paths
# ------------------------------------------------------------------ #
APP_DIR           = Path("/apps/python/patternai")
SESSIONS_DIR      = APP_DIR / "sessions"
FEEDBACK_LOG      = APP_DIR / "bin/feedback.jsonl"
LOG_FILE          = APP_DIR / "bin/app.log"

# ------------------------------------------------------------------ #
#  DB2
# ------------------------------------------------------------------ #
KEYWORDS_CACHE_TTL_SECS = 3600

# ------------------------------------------------------------------ #
#  Flask
# ------------------------------------------------------------------ #
FLASK_HOST        = "0.0.0.0"
FLASK_PORT        = 5000
FLASK_DEBUG       = False

# ------------------------------------------------------------------ #
#  Session management
# ------------------------------------------------------------------ #
SESSION_MAX_BYTES = 10 * 1024   # 10KB per session file
SESSION_TTL_SECS  = 5 * 60     # 5 minutes inactivity before expiry
SESSION_SWEEP_SECS = 60         # How often the cleanup thread runs

# ------------------------------------------------------------------ #
#  Search defaults
# ------------------------------------------------------------------ #
DEFAULT_MAX_RESULTS  = 9
PRESIGN_EXPIRY_SECS  = 3600     # 1 hour for S3 pre-signed URLs
RERANK_FETCH_MULT    = 3        # Fetch this many × max_results when reranking

# ------------------------------------------------------------------ #
#  Bootstrap — point boto3 at our credentials and create dirs
# ------------------------------------------------------------------ #
def bootstrap():
    """Call once at startup to set environment and create required directories."""
    os.environ.setdefault("AWS_SHARED_CREDENTIALS_FILE", str(AWS_CREDS_FILE))
    os.environ.setdefault("AWS_CONFIG_FILE",             str(AWS_CONFIG_FILE))
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    APP_DIR.mkdir(parents=True, exist_ok=True)
