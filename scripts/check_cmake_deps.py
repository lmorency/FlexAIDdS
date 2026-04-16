#!/usr/bin/env python3
"""Check CMake FetchContent dependencies for newer versions.

Parses CMakeLists.txt for FetchContent_Declare blocks, queries the GitHub API
for the latest release of each, and reports outdated dependencies.
When run in CI with GITHUB_TOKEN, can optionally open an issue.

Usage:
    python scripts/check_cmake_deps.py [--cmake PATH] [--create-issue]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

# Map FetchContent name → GitHub repo (owner/name)
KNOWN_DEPS = {
    "nlohmann_json": "nlohmann/json",
    "googletest": "google/googletest",
}


def parse_fetchcontent_versions(cmake_path: str) -> dict[str, tuple[str, str]]:
    """Parse CMakeLists.txt for FetchContent_Declare blocks.

    Returns dict of {dep_name: (git_tag, github_repo_url)}.
    """
    with open(cmake_path, "r", encoding="utf-8") as f:
        content = f.read()

    results = {}
    # Match FetchContent_Declare blocks
    pattern = re.compile(
        r"FetchContent_Declare\s*\(\s*"
        r"(\w+)\s+"
        r"GIT_REPOSITORY\s+(https://github\.com/[\w\-]+/[\w\-]+(?:\.git)?)\s+"
        r"GIT_TAG\s+([\w.\-]+)",
        re.MULTILINE,
    )

    for match in pattern.finditer(content):
        name = match.group(1)
        repo_url = match.group(2)
        tag = match.group(3)
        results[name] = (tag, repo_url)

    return results


def get_latest_release(repo: str) -> str | None:
    """Get the latest release tag from GitHub API."""
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "FlexAIDdS-dep-checker")

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return data.get("tag_name")
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"  Warning: Could not fetch latest release for {repo}: {e}", file=sys.stderr)
        return None


def check_existing_issue(repo: str, title: str) -> bool:
    """Check if an open issue with the given title already exists."""
    url = f"https://api.github.com/repos/{repo}/issues?state=open&per_page=100"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "FlexAIDdS-dep-checker")

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            issues = json.loads(resp.read().decode())
            return any(issue.get("title") == title for issue in issues)
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


def create_issue(repo: str, title: str, body: str) -> bool:
    """Create a GitHub issue. Returns True on success."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("Warning: No GITHUB_TOKEN — cannot create issue", file=sys.stderr)
        return False

    url = f"https://api.github.com/repos/{repo}/issues"
    data = json.dumps({"title": title, "body": body, "labels": ["dependencies"]}).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "FlexAIDdS-dep-checker")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            print(f"Created issue: {result.get('html_url')}")
            return True
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"Warning: Could not create issue: {e}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CMake FetchContent deps for updates")
    parser.add_argument("--cmake", default="CMakeLists.txt", help="Path to CMakeLists.txt")
    parser.add_argument(
        "--create-issue", action="store_true", help="Create GitHub issue if deps are outdated"
    )
    parser.add_argument(
        "--repo", default="LeBonhommePharma/FlexAIDdS", help="GitHub repo for issue creation"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.cmake):
        print(f"Error: {args.cmake} not found", file=sys.stderr)
        return 1

    versions = parse_fetchcontent_versions(args.cmake)
    if not versions:
        print("No FetchContent_Declare blocks found")
        return 0

    print(f"Found {len(versions)} FetchContent dependencies:\n")

    outdated = []
    for name, (current_tag, repo_url) in sorted(versions.items()):
        github_repo = KNOWN_DEPS.get(name)
        if not github_repo:
            # Extract from URL: https://github.com/owner/repo.git → owner/repo
            match = re.match(r"https://github\.com/([\w\-]+/[\w\-]+)", repo_url)
            github_repo = match.group(1) if match else None

        if not github_repo:
            print(f"  {name}: {current_tag} (unknown repo, skipped)")
            continue

        latest = get_latest_release(github_repo)
        if latest is None:
            print(f"  {name}: {current_tag} (could not check latest)")
            continue

        if latest == current_tag:
            print(f"  {name}: {current_tag} (up to date)")
        else:
            print(f"  {name}: {current_tag} -> {latest} (UPDATE AVAILABLE)")
            outdated.append((name, current_tag, latest, github_repo))

    if not outdated:
        print("\nAll dependencies are up to date.")
        return 0

    print(f"\n{len(outdated)} dependency update(s) available.")

    if args.create_issue:
        title = "Update: CMake FetchContent dependencies"
        if check_existing_issue(args.repo, title):
            print("Issue already exists — skipping creation")
        else:
            body_lines = ["The following CMake FetchContent dependencies have newer versions:\n"]
            for name, current, latest, repo in outdated:
                body_lines.append(f"- **{name}** (`{repo}`): `{current}` → `{latest}`")
            body_lines.append("\nUpdate the `GIT_TAG` values in `CMakeLists.txt`.")
            create_issue(args.repo, title, "\n".join(body_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
