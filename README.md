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

### Web dashboard (recommended)

```bash
source .venv/bin/activate
streamlit run app.py
```

Streamlit opens a browser tab. Enter a URL, click **Run review**, and watch progress through the four stages (fetch → parse → link check → streamed Claude review). Past reviews from the current session appear in the sidebar.

### CLI

```bash
source .venv/bin/activate
python seo_review.py https://example.com
```

Either entry point runs the same pipeline:

1. Fetch the page and parse the HTML.
2. Send a HEAD/GET request to every link on the page and record the HTTP status.
3. Send the page metadata, headings, image alt text, link-check results, and body text to Claude (`claude-sonnet-4-6`).
4. Produce a report covering: broken links, grammar/spelling, copywriting suggestions (evaluated against PAS, AIDA, BAB, and StoryBrand), on-page SEO recommendations, and priority fixes.

## Deploy to Streamlit Community Cloud

Free, deploys directly from this GitHub repo, gives you a hosted URL.

1. Push your changes to `main` on GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app** → pick `djgower427/seo-tool` → branch `main` → main file `app.py`.
4. Expand **Advanced settings → Secrets** and paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   (Same format as `.streamlit/secrets.toml.example`.)
5. Click **Deploy**. First build takes ~1–2 minutes while it installs `requirements.txt`.

**Important:** the resulting URL is public by default — anyone who finds it can run reviews and burn your API key. Use **Manage app → Settings → Sharing → Only specific viewers** to gate by email, or treat the URL as a secret.

## Scope (v1)

- Single URL only — no crawling.
- No Semrush data — coming in v2.
- Dashboard history is in-memory per session (lost on restart); no file export yet.
