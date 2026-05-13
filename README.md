# seo-tool

Reviews a single web page for broken links, grammar issues, copywriting improvements, and on-page SEO opportunities. Uses the Anthropic API for analysis.

Semrush integration is planned for v2.

## Setup

```bash
# 1. Clone & enter the repo
cd seo-tool

# 2. Create a virtualenv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Copy .env.example to .env and fill in your keys
cp .env.example .env
# then edit .env — ANTHROPIC_API_KEY is required; SEMRUSH_API_KEY is unused in v1
```

## Run

```bash
source .venv/bin/activate
python seo_review.py https://example.com
```

The script will:

1. Fetch the page and parse the HTML.
2. Send a HEAD/GET request to every link on the page and record the HTTP status.
3. Send the page metadata, headings, image alt text, link-check results, and body text to Claude (`claude-sonnet-4-6`).
4. Print a readable terminal report covering: broken links, grammar/spelling, copywriting suggestions, on-page SEO recommendations, and priority fixes.

## Scope (v1)

- Single URL only — no crawling.
- No Semrush data — coming in v2.
- Output is terminal text; no file export yet.
