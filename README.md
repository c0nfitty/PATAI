<div align="center">

<img src="https://img.shields.io/badge/IBM%20i-PASE-blue?style=for-the-badge&logo=ibm&logoColor=white" alt="IBM i"/>
<img src="https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/AWS-Bedrock-orange?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="AWS Bedrock"/>
<img src="https://img.shields.io/badge/Claude-Sonnet%204-blueviolet?style=for-the-badge" alt="Claude"/>
<img src="https://img.shields.io/badge/Flask-2.2+-green?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>

<br/><br/>

```
  ╔═══════════════════════════════════════════╗
  ║   AI  PATTERN  SEARCH                     ║
  ║   Maple Rugs — Design Team Search Tool    ║
  ╚═══════════════════════════════════════════╝
```

**Natural language rug pattern search powered by AWS Bedrock, Claude, and IBM i DB2**

[Features](#features) · [Architecture](#architecture) · [Setup](#setup) · [Usage](#usage) · [Project Structure](#project-structure)

</div>

---

## Overview

AI Pattern Search is an internal design tool that lets the Maple Rugs design team search a library of 60,000+ rug patterns using natural language. Instead of manually browsing spreadsheets, designers describe what they need — *"something tribal with earth tones, bold geometric, not too traditional"* — and the system finds the closest matches instantly.

Built on **AWS Bedrock Knowledge Bases** for semantic search, **Claude Sonnet** for query expansion and summarisation, and **Flask** running in PASE on **IBM i**.

---

## Features

- 🔍 **Natural language search** — describe mood, colour, style, origin in plain English
- 🧠 **Query expansion** — Claude enriches short queries with design vocabulary before searching
- ⬆️ **Reranking** — Cohere reranker reorders results for higher relevance precision
- 💬 **Conversation mode** — session-aware search that avoids showing the same patterns twice
- 🏷️ **Keyword taxonomy** — 57 design keywords stored in IBM i DB2, served as suggestion chips
- 🚩 **Feedback logging** — design team can flag irrelevant results for pipeline improvement
- 🖼️ **Image grid** — S3 pre-signed image URLs displayed in a dark glassmorphism UI
- 📋 **Full metadata** — style, pattern type, colours, design elements, origin, material per result

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (Design Team PC)                                    │
│  http://s1052yzm.maplesrugs.com:5000                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────┐
│  IBM i — PASE                                                │
│  Flask app  /apps/python/patternai/                         │
│  Python 3.9 · boto3 · ibm_db_dbi                            │
│                          │                                   │
│  ┌───────────┐   ┌───────▼────────┐   ┌──────────────────┐ │
│  │ DB2       │   │ Session files  │   │ Feedback log     │ │
│  │ MFGDBFA   │   │ IFS /sessions/ │   │ feedback.jsonl   │ │
│  │ PATAI_KWDS│   └────────────────┘   └──────────────────┘ │
│  └───────────┘                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTPS (boto3)
┌──────────────────────────▼──────────────────────────────────┐
│  AWS  us-east-1                                              │
│                                                              │
│  ┌─────────────────┐   ┌─────────────────────────────────┐ │
│  │ Bedrock KB      │   │ Anthropic API                   │ │
│  │ MHADZDLOPE      │   │ claude-sonnet-4-6               │ │
│  │ RetrieveAndGen  │   │ Query expansion · Summarisation │ │
│  │ Cohere Rerank   │   └─────────────────────────────────┘ │
│  └────────┬────────┘                                        │
│           │                                                  │
│  ┌────────▼────────────────────────────────────────────┐   │
│  │ S3                                                   │   │
│  │ variety-bucket-514316422605-us-east-1-an  (images)  │   │
│  │ result-buckett  (JSON metadata)                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
/apps/python/patternai/
│
├── app.py              # Flask routes — thin layer, no business logic
├── config.py           # All constants, paths, and env vars
├── search.py           # Bedrock retrieval, S3 presigning, Cohere reranking
├── rag.py              # Query expansion and RAG summary (Claude + Bedrock)
├── session.py          # Conversation mode — IFS session file management
├── keywords.py         # DB2-backed keyword taxonomy with 1hr cache
├── feedback.py         # Flagged result logging to JSONL
│
├── templates/
│   └── index.html      # Dark glassmorphism UI (HTML/CSS/JS)
│
├── aws/
│   ├── credentials     # AWS credentials (not committed to git)
│   └── config          # AWS region config
│
├── sessions/           # Conversation session files (auto-cleaned)
├── start.sh            # PASE startup script
├── stop.sh             # Graceful stop via PID file
├── requirements.txt    # pip dependencies
└── feedback.jsonl      # Flagged result log (runtime, not committed)
```

---

## Setup

### Prerequisites

| Requirement | Version |
|---|---|
| IBM i | V7R4+ with PASE enabled |
| Python | 3.9+ (via YUM) |
| AWS Account | Bedrock enabled in us-east-1 |
| Anthropic API Key | console.anthropic.com |

### 1. Install dependencies

```sh
# In PASE bash on IBM i
export PATH=/QOpenSys/pkgs/bin:$PATH
pip3 install -r requirements.txt
yum install python3-ibm_db
```

### 2. Configure environment

Copy AWS credentials to the app directory:
```sh
mkdir -p /apps/python/patternai/aws
# copy credentials and config files from your workstation
```

Set your Anthropic API key in `start.sh`:
```sh
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Set up DB2 keyword table

```sql
CREATE OR REPLACE TABLE MFGDBFA.PATAI_KWDS (
    keywordID  FOR COLUMN KWID    CHAR(36)     NOT NULL DEFAULT '',
    keyword    FOR COLUMN KEYW    VARCHAR(50)  NOT NULL,
    statusCode FOR COLUMN STSCDE  SMALLINT     NOT NULL DEFAULT 1,
    creationDate FOR COLUMN CRTDATE TIMESTAMP  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    creationUser FOR COLUMN CRTUSER VARCHAR(25) NOT NULL DEFAULT USER,
    CONSTRAINT PATAI_KWDS_PK PRIMARY KEY (KWID)
)
RCDFMT PATAI_KWWR;
```

### 4. Start the application

```sh
/apps/python/patternai/start.sh
```

Access at: `http://your-ibmi-hostname:5000`

### 5. Automatic startup (IBM i subsystem)

Add an autostart job entry to your web subsystem:
```
ADDAJE SBSD(QSYS/WEBSBS3) JOB(PATAISTR) JOBD(QGPL/PATAIJOBD)
```

---

## Usage

### Search

Type a natural language description into the search bar:

> *"bold geometric in navy and terracotta, tribal kilim, stepped border"*

> *"something soft and transitional for a bedroom, blush tones, floral"*

> *"coastal, faded, sandy beige with hints of ocean blue, relaxed"*

### Options

| Toggle | Description |
|---|---|
| **Query expansion** | Claude enriches your query with design vocabulary before searching |
| **Rerank results** | Cohere reranker reorders results for higher relevance |
| **Conversation mode** | Session-aware — won't show the same patterns twice |
| **Results** | 9 / 18 / 27 / 36 results per search |

### Keyword chips

Click any keyword chip for a one-click search. Hit **↻** to load a fresh random selection from the DB2 keyword table.

### Flagging results

Click the 🚩 flag on any card to mark it as not relevant. Flags are logged to `feedback.jsonl` for pipeline review.

---

## Data Pipeline

Rug metadata is generated by a separate vision pipeline (`/pipeline`) that:

1. Reads PNG images from S3
2. Sends each image to Claude via AWS Bedrock with a structured analysis prompt
3. Writes JSON metadata files back to S3
4. The Bedrock Knowledge Base indexes the JSON files for semantic search

To improve search quality after pipeline changes, run `rewrite_combined_text.py` to rebuild the `combined_text` field without re-running the vision model, then re-sync the Knowledge Base.

---

## AWS Resources

| Resource | ID / Name |
|---|---|
| Knowledge Base | `MHADZDLOPE` |
| Image bucket | `variety-bucket-514316422605-us-east-1-an` |
| JSON bucket | `result-buckett` |
| JSON prefix | `aws/bedrock/knowledge_bases/ASC-VAR/` |
| Bedrock model | `us.anthropic.claude-sonnet-4-6` |
| Rerank model | `cohere.rerank-v3-5:0` |
| Region | `us-east-1` |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude |
| `AWS_SHARED_CREDENTIALS_FILE` | Auto-set | Points to `/apps/python/patternai/aws/credentials` |
| `AWS_CONFIG_FILE` | Auto-set | Points to `/apps/python/patternai/aws/config` |

---

## Contributing

This is an internal Maple Rugs project. To add keywords to the search taxonomy, update `MFGDBFA.PATAI_KWDS` directly — the app picks up changes within 1 hour (cache TTL).

To improve retrieval quality, review `feedback.jsonl` periodically and update the pipeline prompt in `src/maplerugs/agents/description_agent.py`.

---

<div align="center">

Built with ☕ by the Maple Rugs technology team

*IBM i · AWS Bedrock · Claude · Python · Flask*

</div>
