"""Relative links in README.md and docs/*.md point at files and headings that exist."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from .conftest import REPO

DOCS = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]
LINK = re.compile(r"\]\(([^)\s]+)\)")


def slug(heading: str) -> str:
    """GitHub's heading anchor: lowercase, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)  # no headings in code blocks
    return {slug(m.group(1)) for m in re.finditer(r"^#+ (.+)$", text, flags=re.M)}


def test_docs_exist():
    names = {p.name for p in DOCS}
    assert {"simulation.md", "real-robot.md", "input-devices.md", "safety.md"} <= names


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_relative_links_resolve(doc):
    text = re.sub(r"```.*?```", "", doc.read_text(), flags=re.S)
    for target in LINK.findall(text):
        if re.match(r"[a-z]+:", target):  # http(s), mailto: not checked here
            continue
        file, _, anchor = target.partition("#")
        dest = (doc.parent / file).resolve() if file else doc
        assert dest.exists(), f"{doc.name}: {target} -> missing {dest}"
        if anchor:
            assert anchor in anchors(dest), f"{doc.name}: {target} -> no heading #{anchor}"


def test_readme_contents_lists_every_section():
    """The README's table of contents names every ## / ### heading, in order."""
    text = re.sub(r"```.*?```", "", (REPO / "README.md").read_text(), flags=re.S)
    headings = [slug(h) for h in re.findall(r"^#{2,3} (.+)$", text, flags=re.M)]
    toc = text.split("## Contents", 1)[1].split("\n## ", 1)[0]
    listed = re.findall(r"\]\(#([^)]+)\)", toc)
    assert listed == [h for h in headings if h != "contents"]
