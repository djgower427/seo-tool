"""SEO review tool — fetch a URL, analyze it, print a report."""

import argparse
import os
import sys
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"
USER_AGENT = "seo-review-tool/0.1 (+https://github.com/)"
REQUEST_TIMEOUT = 15.0
LINK_CHECK_TIMEOUT = 10.0
MAX_BODY_CHARS = 12000


def fetch_page(url: str) -> tuple[str, str]:
    """Fetch a page and return (final_url, html). Raises on failure."""
    with httpx.Client(
        follow_redirects=True,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return str(resp.url), resp.text


def extract_page_data(final_url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    meta_desc = ""
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        meta_desc = md["content"].strip()

    canonical = ""
    can = soup.find("link", attrs={"rel": "canonical"})
    if can and can.get("href"):
        canonical = can["href"].strip()

    robots = ""
    rm = soup.find("meta", attrs={"name": "robots"})
    if rm and rm.get("content"):
        robots = rm["content"].strip()

    headings = {f"h{i}": [h.get_text(strip=True) for h in soup.find_all(f"h{i}")] for i in range(1, 4)}

    images = []
    for img in soup.find_all("img"):
        images.append({"src": img.get("src", ""), "alt": img.get("alt", "")})

    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(final_url, href)
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append({"href": absolute, "text": a.get_text(strip=True)[:120]})

    # Visible-ish text body for grammar/copy review
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    body_text = " ".join(soup.get_text(separator=" ").split())

    return {
        "final_url": final_url,
        "title": title,
        "meta_description": meta_desc,
        "canonical": canonical,
        "robots": robots,
        "headings": headings,
        "images": images,
        "links": links,
        "body_text": body_text,
        "word_count": len(body_text.split()),
    }


def check_links(
    links: list[dict],
    base_url: str,
    on_progress=None,
) -> list[dict]:
    """HEAD-check each link; fall back to GET if HEAD isn't supported.

    on_progress(i, total, href) is called before each request, when provided.
    """
    results = []
    base_host = urlparse(base_url).netloc
    total = len(links)
    with httpx.Client(
        follow_redirects=True,
        timeout=LINK_CHECK_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for i, link in enumerate(links, 1):
            href = link["href"]
            if on_progress is not None:
                on_progress(i, total, href)
            status: int | None = None
            error = ""
            try:
                r = client.head(href)
                if r.status_code in (405, 403, 501) or r.status_code >= 400:
                    # Some servers reject HEAD — retry with GET
                    r = client.get(href)
                status = r.status_code
            except httpx.HTTPError as e:
                error = f"{type(e).__name__}: {e}"
            results.append({
                "href": href,
                "text": link["text"],
                "status": status,
                "error": error,
                "internal": urlparse(href).netloc == base_host,
            })
    return results


def build_prompt(page: dict, link_results: list[dict]) -> str:
    broken = [r for r in link_results if r["error"] or (r["status"] is not None and r["status"] >= 400)]
    ok = [r for r in link_results if not r["error"] and r["status"] is not None and r["status"] < 400]

    lines = [
        f"URL: {page['final_url']}",
        f"Title: {page['title']!r}  (length: {len(page['title'])})",
        f"Meta description: {page['meta_description']!r}  (length: {len(page['meta_description'])})",
        f"Canonical: {page['canonical'] or '(none)'}",
        f"Robots: {page['robots'] or '(none)'}",
        f"Word count: {page['word_count']}",
        "",
        "Headings:",
    ]
    for level in ("h1", "h2", "h3"):
        hs = page["headings"][level]
        lines.append(f"  {level} ({len(hs)}): {hs}")

    lines.append("")
    lines.append(f"Images: {len(page['images'])} total, "
                 f"{sum(1 for i in page['images'] if not i['alt'])} missing alt text")
    for img in page["images"][:20]:
        lines.append(f"  - src={img['src']!r} alt={img['alt']!r}")

    lines.append("")
    lines.append(f"Link check results: {len(link_results)} links, {len(broken)} broken/errored, {len(ok)} ok")
    if broken:
        lines.append("Broken/errored links:")
        for r in broken:
            status = r["status"] if r["status"] is not None else f"ERROR ({r['error']})"
            lines.append(f"  - [{status}] {r['href']}  text={r['text']!r}")
    lines.append("Working links (sample):")
    for r in ok[:15]:
        lines.append(f"  - [{r['status']}] {r['href']}")

    body = page["body_text"][:MAX_BODY_CHARS]
    truncated = " …[truncated]" if len(page["body_text"]) > MAX_BODY_CHARS else ""
    lines.append("")
    lines.append("Page body text (for grammar/copy review):")
    lines.append('"""')
    lines.append(body + truncated)
    lines.append('"""')

    summary = "\n".join(lines)

    instructions = """\
You are an experienced SEO consultant and copy editor reviewing a single web page.

Using ONLY the data provided below (do not invent details about the page that aren't shown), produce a readable terminal report with these sections, in this order:

1. **Summary** — 2-3 sentences on the page's overall health.
2. **Broken Links** — list each broken/errored link with its status and a brief note. If none, say so.
3. **Grammar & Spelling** — specific issues you spot in the body text, with the offending phrase quoted.
4. **Copywriting Suggestions** — evaluate the page against these established copywriting frameworks and give concrete improvements:
   - **PAS (Problem–Agitate–Solution)**: Does the page name a problem the reader has, agitate the pain of it, and present the offering as the solution? Quote what's working and what's missing.
   - **AIDA (Attention–Interest–Desire–Action)**: Does the headline/hero grab attention, the next section build interest, the body create desire, and a clear CTA drive action? Identify which stage is weakest.
   - **BAB (Before–After–Bridge)**: Does the copy paint the reader's "before" state, the "after" state they want, and bridge with the offering? Suggest specific rewrites if absent.
   - **StoryBrand (SB7)**: Does the page cast the reader as the hero with a clear problem, position the brand as the guide with a plan, call them to action, and contrast success vs. failure? Flag any of the 7 elements that are missing or muddled.
   For each framework, give a 1-line verdict (e.g. "PAS: weak — problem is implied but never named") plus 1–2 concrete rewrites or additions. End the section with a short "Recommended primary framework" pick based on the page's intent and what's already on the page.
5. **On-Page SEO** — title tag, meta description, headings hierarchy, image alt text, internal linking, content depth. Be specific and actionable.
6. **Priority Fixes** — a numbered list of the top 3-5 things to address first.

Use plain text with light Markdown (headings, bullets). Keep it scannable. Don't pad — every bullet should be actionable or specific.
"""
    return instructions + "\n---\nPAGE DATA:\n\n" + summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an SEO review on a single URL.")
    parser.add_argument("url", help="The page URL to review (e.g. https://example.com)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set. Add it to .env.", file=sys.stderr)
        return 1

    url = args.url
    if not urlparse(url).scheme:
        url = "https://" + url

    print(f"Fetching {url} …", file=sys.stderr)
    try:
        final_url, html = fetch_page(url)
    except httpx.HTTPError as e:
        print(f"ERROR fetching page: {e}", file=sys.stderr)
        return 1

    print(f"Parsing page (final URL: {final_url}) …", file=sys.stderr)
    page = extract_page_data(final_url, html)
    print(f"  found {len(page['links'])} unique links, {len(page['images'])} images, "
          f"{page['word_count']} words", file=sys.stderr)

    print(f"Checking {len(page['links'])} links …", file=sys.stderr)

    def _cli_progress(i: int, total: int, href: str) -> None:
        print(f"  [{i}/{total}] {href}", file=sys.stderr)

    link_results = check_links(page["links"], final_url, on_progress=_cli_progress)

    print("Calling Claude for the review …", file=sys.stderr)
    client = Anthropic(api_key=api_key)
    prompt = build_prompt(page, link_results)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    report = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")

    print()
    print("=" * 72)
    print(f"SEO Review: {final_url}")
    print("=" * 72)
    print()
    print(report)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
