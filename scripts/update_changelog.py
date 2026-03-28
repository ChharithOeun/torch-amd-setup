#!/usr/bin/env python3
"""
update_changelog.py — Auto-update CHANGELOG.md from git history.

Run this after committing (or wire into a git post-commit hook):
    python scripts/update_changelog.py

What it does:
  1. Reads git log since the last changelog update commit
  2. Categorises commits by conventional commit prefix (feat/fix/docs/chore/...)
  3. Injects a new [Unreleased] block at the top of CHANGELOG.md
  4. Commits the result (optional — pass --commit to auto-commit)

Usage:
    python scripts/update_changelog.py           # update only
    python scripts/update_changelog.py --commit  # update + git commit
    python scripts/update_changelog.py --dry-run # print what would change

Policy: ZERO fabricated numbers. Only commit messages go into the changelog.
"""

import subprocess
import re
import datetime
import sys
import argparse
from pathlib import Path

REPO_ROOT  = Path(__file__).parent.parent
CHANGELOG  = REPO_ROOT / "CHANGELOG.md"

CATEGORY_MAP = {
    "feat":     "### Added",
    "fix":      "### Fixed",
    "docs":     "### Changed",
    "chore":    "### Changed",
    "refactor": "### Changed",
    "perf":     "### Added",
    "test":     "### Added",
    "security": "### Security",
    "sec":      "### Security",
}


def git(*args) -> str:
    result = subprocess.run(["git"] + list(args), capture_output=True, text=True,
                            cwd=REPO_ROOT)
    return result.stdout.strip()


def get_new_commits() -> list[dict]:
    """Return commits since the last time CHANGELOG.md was touched."""
    last_changelog_sha = git("log", "--follow", "--pretty=format:%H",
                             "--", "CHANGELOG.md").splitlines()

    if last_changelog_sha:
        log_range = f"{last_changelog_sha[0]}..HEAD"
    else:
        log_range = "HEAD~30..HEAD"

    raw = git("log", log_range, "--pretty=format:%h|%ad|%s|%an", "--date=short")
    if not raw:
        return []

    commits = []
    for line in raw.splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            sha, date, subject, author = parts
            commits.append({"sha": sha, "date": date, "subject": subject, "author": author})
    return commits


def categorise(commits: list[dict]) -> dict[str, list[str]]:
    """Categorise commit subjects by conventional commit prefix."""
    buckets: dict[str, list[str]] = {}

    for c in commits:
        subject = c["subject"]
        sha     = c["sha"]

        m = re.match(r'^(feat|fix|docs|chore|refactor|perf|test|sec|security)(\([^)]+\))?!?:\s*', subject)
        prefix = m.group(1) if m else "chore"
        header = CATEGORY_MAP.get(prefix, "### Changed")

        scope_m = re.match(r'^\w+\(([^)]+)\)', subject)
        scope   = f"**{scope_m.group(1)}** — " if scope_m else ""

        body = re.sub(r'^\w+(\([^)]+\))?!?:\s*', '', subject)

        if header not in buckets:
            buckets[header] = []
        buckets[header].append(f"- {scope}{body} (`{sha}`)")

    return buckets


def build_unreleased_block(buckets: dict[str, list[str]], n_commits: int) -> str:
    today = datetime.date.today().isoformat()
    lines = [f"## [Unreleased] — updated {today} ({n_commits} new commit(s))", ""]

    preferred_order = ["### Added", "### Fixed", "### Changed", "### Security"]
    for header in preferred_order:
        if header in buckets:
            lines.append(header)
            lines.extend(buckets[header])
            lines.append("")

    return "\n".join(lines) + "\n"


def inject_into_changelog(new_block: str) -> str:
    """Replace the existing [Unreleased] section (or insert before first versioned section)."""
    content = CHANGELOG.read_text(encoding="utf-8")

    content = re.sub(
        r'## \[Unreleased\][^\n]*\n(?:(?!## \[).*\n)*',
        '',
        content,
        flags=re.MULTILINE
    )

    match = re.search(r'^## \[', content, re.MULTILINE)
    if match:
        pos = match.start()
        content = content[:pos] + new_block + "\n" + content[pos:]
    else:
        content = content.rstrip() + "\n\n" + new_block

    return content


def main():
    parser = argparse.ArgumentParser(description="Auto-update CHANGELOG.md")
    parser.add_argument("--commit",  action="store_true", help="Auto-commit the changelog update")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    commits = get_new_commits()
    if not commits:
        print("✅ CHANGELOG up to date — no new commits.")
        return

    print(f"📋 {len(commits)} new commit(s) to add:")
    for c in commits:
        print(f"   {c['sha']} {c['subject']}")

    buckets      = categorise(commits)
    new_block    = build_unreleased_block(buckets, len(commits))
    new_content  = inject_into_changelog(new_block)

    if args.dry_run:
        print("\n--- Would write to CHANGELOG.md ---")
        print(new_block)
        return

    CHANGELOG.write_text(new_content, encoding="utf-8")
    print(f"✅ CHANGELOG.md updated.")

    if args.commit:
        subprocess.run(["git", "add", str(CHANGELOG)], cwd=REPO_ROOT)
        subprocess.run(
            ["git", "commit", "-m", "chore(changelog): auto-update [skip ci]"],
            cwd=REPO_ROOT
        )
        print("✅ Changelog committed.")


if __name__ == "__main__":
    main()
