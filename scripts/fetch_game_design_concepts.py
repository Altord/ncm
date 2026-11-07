#!/usr/bin/env python3
"""Fetch Ian Schreiber's CC BY 3.0 Game Design Concepts posts via WordPress API."""

from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen


API = ("https://public-api.wordpress.com/rest/v1.1/sites/"
       "gamedesignconcepts.wordpress.com/posts/?number=100&_fields=found,posts")
ATTRIBUTION = """# Source and license

Ian Schreiber, *Game Design Concepts* (2009), retrieved from
https://gamedesignconcepts.wordpress.com/ . Licensed CC BY 3.0 US:
https://creativecommons.org/licenses/by/3.0/us/ . The material may be shared
and adapted with attribution. Original post title, URL, and publication date
are retained in each file. This local copy was fetched by
`scripts/fetch_game_design_concepts.py`.
"""


class TextFromHTML(HTMLParser):
    BLOCKS = {"p", "div", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6",
              "li", "ul", "ol", "blockquote", "pre", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCKS:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag):
        if tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)

    def text(self):
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        output = []
        for line in lines:
            if line or (output and output[-1]):
                output.append(line)
        return "\n".join(output).strip()


def main():
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "data/reference/game-design-concepts")
    destination.mkdir(parents=True, exist_ok=True)
    request = Request(API, headers={"User-Agent": "ncm-reference-sync/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    posts = sorted(payload.get("posts", []), key=lambda post: (post.get("date", ""), post["ID"]))
    if not posts:
        raise SystemExit("WordPress API returned no posts")
    for post in posts:
        parser = TextFromHTML()
        parser.feed(post.get("content", ""))
        body = parser.text()
        title = post.get("title", "Untitled")
        filename = f"{post['ID']}.md"
        text = f"# {title}\n\nSource: {post.get('URL', '')}\nPublished: {post.get('date', '')}\n\n{body}\n"
        (destination / filename).write_text(text, encoding="utf-8")
    (destination / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")
    metadata = {
        "source": "Ian Schreiber, Game Design Concepts",
        "api": API,
        "license": "CC BY 3.0 US",
        "license_url": "https://creativecommons.org/licenses/by/3.0/us/",
        "posts": len(posts),
        "post_ids": [post["ID"] for post in posts],
    }
    (destination / "SOURCE.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Fetched {len(posts)} posts to {destination}")


if __name__ == "__main__":
    main()
