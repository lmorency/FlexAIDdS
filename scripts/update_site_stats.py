#!/usr/bin/env python3
"""Update hardcoded repository stats in site/index.html.

Reads commit count from git and language breakdown from GitHub API,
then patches site/index.html in-place. Uses only the standard library.

Usage:
    python scripts/update_site_stats.py [--repo OWNER/REPO] [--html PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

# GitHub language name → (CSS variable suffix, short display name)
LANG_MAP = {
    "C++": ("cpp", "C++"),
    "Python": ("python", "Python"),
    "Swift": ("swift", "Swift"),
    "Objective-C++": ("objcpp", "Obj-C++"),
    "CMake": ("cmake", "CMake"),
    "TypeScript": ("ts", "TypeScript"),
    "Cuda": ("cuda", "CUDA"),
    "CUDA": ("cuda", "CUDA"),
    "C": ("c", "C"),
    "Shell": ("other", "Shell"),
    "Metal": ("other", "Metal"),
    "JavaScript": ("other", "JavaScript"),
}

MIN_PERCENT = 1.0  # Languages below this are grouped into "Other"


def get_commit_count() -> int:
    """Return total commit count on the current branch."""
    result = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def fetch_languages(repo: str) -> dict[str, int]:
    """Fetch language byte counts from GitHub API."""
    url = f"https://api.github.com/repos/{repo}/languages"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "FlexAIDdS-site-updater")

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"Warning: Could not fetch languages from GitHub API: {e}", file=sys.stderr)
        return {}


def compute_percentages(languages: dict[str, int]) -> list[tuple[str, str, float]]:
    """Compute (css_var_suffix, display_name, percentage) sorted by percentage desc.

    Languages below MIN_PERCENT are grouped into 'Other'.
    """
    total = sum(languages.values())
    if total == 0:
        return []

    entries: list[tuple[str, str, float]] = []
    other_pct = 0.0

    for lang, bytes_count in sorted(languages.items(), key=lambda x: -x[1]):
        pct = round(bytes_count / total * 100, 1)
        if lang in LANG_MAP:
            css_suffix, display = LANG_MAP[lang]
            if pct < MIN_PERCENT:
                other_pct += pct
            else:
                entries.append((css_suffix, display, pct))
        else:
            other_pct += pct

    if other_pct > 0:
        entries.append(("other", "Other", round(other_pct, 1)))

    return entries


def build_lang_bar(entries: list[tuple[str, str, float]]) -> str:
    """Build the lang-bar HTML block."""
    lines = ['        <div class="lang-bar" aria-label="Language breakdown">']
    for css_suffix, display, pct in entries:
        lines.append(
            f'          <div class="lang-segment" style="width:{pct}%;'
            f'background:var(--lang-{css_suffix})" title="{display} {pct}%"></div>'
        )
    lines.append("        </div>")
    return "\n".join(lines)


def build_lang_legend(entries: list[tuple[str, str, float]]) -> str:
    """Build the lang-legend HTML block."""
    lines = ['        <div class="lang-legend">']
    for css_suffix, display, pct in entries:
        lines.append(
            f'          <span><i style="background:var(--lang-{css_suffix})"></i>'
            f"{display} {pct}%</span>"
        )
    lines.append("        </div>")
    return "\n".join(lines)


def update_html(html_path: str, commit_count: int, lang_entries: list[tuple[str, str, float]]) -> bool:
    """Patch the HTML file in-place. Returns True if changes were made."""
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    original = content

    # 1. Update commit count: data-count="NNN"
    content = re.sub(
        r'data-count="\d+"',
        f'data-count="{commit_count}"',
        content,
        count=1,
    )

    # 2. Update language count stat card
    if lang_entries:
        n_langs = len(lang_entries)
        # Match: <span class="stat-value">N</span>\n            <span class="stat-label">Languages</span>
        content = re.sub(
            r'(<span class="stat-value">)\d+(</span>\s*<span class="stat-label">Languages</span>)',
            rf"\g<1>{n_langs}\g<2>",
            content,
            count=1,
        )

    # 3. Replace lang-bar block (including all nested divs)
    if lang_entries:
        new_bar = build_lang_bar(lang_entries)
        content = re.sub(
            r' *<div class="lang-bar"[^>]*>.*?</div>\n *</div>',
            new_bar,
            content,
            count=1,
            flags=re.DOTALL,
        )

    # 4. Replace lang-legend block (including all nested spans)
    if lang_entries:
        new_legend = build_lang_legend(lang_entries)
        content = re.sub(
            r' *<div class="lang-legend">.*?</div>',
            new_legend,
            content,
            count=1,
            flags=re.DOTALL,
        )

    changed = content != original

    if changed:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(content)

    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Update site stats in index.html")
    parser.add_argument("--repo", default="lmorency/FlexAIDdS", help="GitHub repo (owner/name)")
    parser.add_argument("--html", default="site/index.html", help="Path to index.html")
    args = parser.parse_args()

    # Get commit count
    try:
        commit_count = get_commit_count()
        print(f"Commit count: {commit_count}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error getting commit count: {e}", file=sys.stderr)
        return 1

    # Get language breakdown
    languages = fetch_languages(args.repo)
    lang_entries = compute_percentages(languages) if languages else []
    if lang_entries:
        print("Language breakdown:")
        for _, display, pct in lang_entries:
            print(f"  {display}: {pct}%")
    else:
        print("Skipping language update (API unavailable)")

    # Update HTML
    if not os.path.isfile(args.html):
        print(f"Error: {args.html} not found", file=sys.stderr)
        return 1

    changed = update_html(args.html, commit_count, lang_entries)
    if changed:
        print(f"Updated {args.html}")
    else:
        print("No changes needed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
