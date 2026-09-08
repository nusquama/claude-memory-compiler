#!/usr/bin/env python3
"""okf_wiki.py — Turn a folder of Obsidian YouTube summaries into an OKF v0.1 bundle.

Conformant to the official Open Knowledge Format v0.1 spec
(GoogleCloudPlatform/knowledge-catalog/okf/SPEC.md):

  - Bundle root = the `Youtube Summary/` folder itself (self-contained, §3).
  - Every concept .md has a `type` frontmatter field (required, §4.1/§9).
  - Cross-links are STANDARD MARKDOWN links, relative, `<>`-wrapped when the path
    has spaces/parens (§5) — NOT `[[wikilinks]]`. Works in Obsidian AND for an
    OKF consumer agent.
  - `resource:` holds the canonical URI (the YouTube URL), §4.1.
  - `index.md` follows §6 (no frontmatter except `okf_version` at the bundle root).

Two effects:
  1. ENRICH IN PLACE — the 168 summaries get OKF frontmatter + a `## Liens` block
     of markdown links to the entities/channel they mention. Body preserved.
  2. NAV CONCEPTS — `_entities/<slug>.md` and `_channels/<slug>.md`, each listing
     its videos (with one-line descriptions), plus a root `index.md`.

Deterministic, no LLM, $0. Entities matched from the curated ENTITY_ALIASES dict.
Helpers slugify/first_paragraph/parse_frontmatter are self-contained (adapted from
CMC utils.py) to stay decoupled from CMC's per-project config.

    python3 scripts/okf_wiki.py --dry-run     # preview, writes nothing
    python3 scripts/okf_wiki.py               # enrich in place + build bundle
    python3 scripts/okf_wiki.py --lint        # check markdown links resolve
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Bundle paths ──────────────────────────────────────────────────────
VAULT = Path(
    "/Users/franck/Library/CloudStorage/Dropbox/Applications/remotely-save/Franck"
)
SRC_DIR = VAULT / "Youtube Summary"          # <- OKF bundle root
OKF_VERSION = "0.1"


def ent_dir() -> Path:
    return SRC_DIR / "_entities"


def chan_dir() -> Path:
    return SRC_DIR / "_channels"


# ── Curated entity dictionary (editable) ──────────────────────────────
ENTITY_ALIASES: dict[str, list[str]] = {
    "OpenClaw": ["openclaw", "open claw"],
    "Hermes": ["hermes"],
    "Claude Code": ["claude code", "claude-code"],
    "Codex": ["codex"],
    "MCP": ["mcp", "model context protocol"],
    "n8n": ["n8n"],
    "Supabase": ["supabase"],
    "Anthropic": ["anthropic"],
    "OpenAI": ["openai", "open ai"],
    "ChatGPT": ["chatgpt", "chat gpt"],
    "Gemini": ["gemini"],
    "Notion": ["notion"],
    "Obsidian": ["obsidian"],
    "Telegram": ["telegram"],
    "Discord": ["discord"],
    "Zapier": ["zapier"],
    "Cursor": ["cursor"],
    "Vercel": ["vercel"],
    "gbrain": ["gbrain"],
    "VPS": ["vps"],
    "Hetzner": ["hetzner"],
    "Railway": ["railway"],
    "Cloudflare": ["cloudflare"],
    "LangChain": ["langchain"],
    "RAG": ["rag"],
    "sub-agent": ["sub-agent", "subagent", "sous-agent"],
    "cron": ["cron"],
    "heartbeat": ["heartbeat"],
}


# ── Tiny helpers ──────────────────────────────────────────────────────
def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-") or "untitled"


def mdlink(text: str, source: Path, target: Path) -> str:
    """A CommonMark relative link from `source` file to `target` file.
    Wraps the destination in <> when it contains spaces or parens (CommonMark §6.6)
    so it stays a single valid link. Works in Obsidian and OKF consumers."""
    dest = os.path.relpath(target, start=source.parent)
    if re.search(r"[ ()<>]", dest):
        dest = f"<{dest}>"
    return f"[{text}]({dest})"


# ── Minimal frontmatter parser ────────────────────────────────────────
def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw = text[3:end].strip("\n")
    body = text[end + 4:].lstrip("\n")
    meta: dict = {}
    lines = raw.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        m = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key, val = m.group(1), m.group(2).strip()
        if val == "":
            items = []
            j = i + 1
            while j < len(lines) and re.match(r"^\s*-\s+", lines[j]):
                items.append(re.sub(r"^\s*-\s+", "", lines[j]).strip().strip('"\''))
                j += 1
            if items:
                meta[key] = items
                i = j
                continue
            meta[key] = ""
        elif val.startswith("[") and val.endswith("]"):
            meta[key] = [x.strip().strip('"\'') for x in val[1:-1].split(",") if x.strip()]
        else:
            meta[key] = val.strip('"\'')
        i += 1
    return meta, body


def yaml_dump_frontmatter(meta: dict) -> str:
    lines = ["---"]
    for key, val in meta.items():
        if isinstance(val, list):
            if not val:
                continue
            lines.append(f"{key}:")
            for item in val:
                s = str(item)
                if s[:1] in "[{#&*!|>%@`\"'" or ":" in s:
                    s = '"' + s.replace('"', '\\"') + '"'
                lines.append(f"  - {s}")
        else:
            s = str(val)
            if s == "":
                continue
            if key in ("title", "description") or ":" in s or s[:1] in "[{#&*!|>%@`\"'":
                s = '"' + s.replace('"', '\\"') + '"'
            lines.append(f"{key}: {s}")
    lines.append("---")
    return "\n".join(lines)


# ── Content helpers ───────────────────────────────────────────────────
def first_paragraph(body: str, limit: int = 200) -> str:
    for para in re.split(r"\n\s*\n", body):
        p = para.strip()
        if not p or p.startswith("!["):
            continue
        is_heading = p.startswith("#")
        p = re.sub(r"^#+\s*", "", p)
        p = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", p)
        p = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", p)
        p = re.sub(r"[*_`>]", "", p)
        p = re.sub(r"\s+", " ", p).strip()
        if is_heading and "." not in p and len(p) < 60:
            continue
        if len(p) > 20:
            return (p[:limit].rsplit(" ", 1)[0] + "…") if len(p) > limit else p
    return ""


def match_entities(text: str) -> list[str]:
    found = []
    for name, aliases in ENTITY_ALIASES.items():
        for alias in aliases:
            if re.search(r"(?<![\w])" + re.escape(alias) + r"(?![\w])", text, re.IGNORECASE):
                found.append(name)
                break
    return found


def strip_liens(body: str) -> str:
    return re.sub(r"\n#+\s*Liens\s*\n.*$", "", body, flags=re.S).rstrip()


# ── Source model ──────────────────────────────────────────────────────
class Video:
    def __init__(self, src: Path):
        self.src = src
        text = src.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        self.body = strip_liens(body)
        self.title = meta.get("title") or src.stem
        self.resource = meta.get("resource") or meta.get("url", "")
        chan = meta.get("channel")
        if not chan:
            chan = src.parent.name if src.parent != SRC_DIR else "Unknown"
        self.channel = chan
        pub = (meta.get("published") or meta.get("created")
               or str(meta.get("timestamp", ""))[:10])
        if not pub:
            pub = datetime.fromtimestamp(src.stat().st_mtime).strftime("%Y-%m-%d")
        self.published = pub
        self.description = first_paragraph(self.body)
        self.entities = match_entities(f"{self.title}\n{self.body}")


# ── Rendering ─────────────────────────────────────────────────────────
def enriched_text(v: Video) -> str:
    tags = list(dict.fromkeys(["youtube", "video-summary"] + [slugify(e) for e in v.entities]))
    meta = {
        "type": "Video Summary",
        "title": v.title,
        "description": v.description,
        "resource": v.resource,
        "channel": v.channel,
        "tags": tags,
        "timestamp": f"{v.published}T00:00:00Z",
    }
    links = [mdlink(v.channel, v.src, chan_dir() / f"{slugify(v.channel)}.md")]
    links += [mdlink(e, v.src, ent_dir() / f"{slugify(e)}.md") for e in v.entities]
    liens = "\n".join(f"- {l}" for l in links)
    return f"{yaml_dump_frontmatter(meta)}\n{v.body}\n\n## Liens\n{liens}\n"


def render_entity(name: str, videos: list[Video], now: str) -> str:
    self_path = ent_dir() / f"{slugify(name)}.md"
    matched = sorted((v for v in videos if name in v.entities),
                     key=lambda v: v.published, reverse=True)
    meta = {
        "type": "Entity", "title": name,
        "description": f"{name} — entité récurrente de la veille YouTube ({len(matched)} vidéos).",
        "tags": ["entity", "tool"], "timestamp": f"{now}T00:00:00Z",
    }
    lines = []
    for v in matched:
        link = mdlink(v.title, self_path, v.src)
        desc = f" - {v.description}" if v.description else ""
        lines.append(f"* {link} — {v.channel}, {v.published}{desc}")
    return (yaml_dump_frontmatter(meta)
            + f"\n# {name}\n\n{name} apparaît dans {len(matched)} vidéo(s).\n\n"
            + f"# Vidéos ({len(matched)})\n" + "\n".join(lines) + "\n")


def render_channel(name: str, videos: list[Video], now: str) -> str:
    self_path = chan_dir() / f"{slugify(name)}.md"
    matched = sorted((v for v in videos if v.channel == name),
                     key=lambda v: v.published, reverse=True)
    meta = {
        "type": "Channel", "title": name,
        "description": f"Chaîne YouTube — {len(matched)} résumé(s) archivé(s).",
        "tags": ["channel", "youtube"], "timestamp": f"{now}T00:00:00Z",
    }
    lines = []
    for v in matched:
        link = mdlink(v.title, self_path, v.src)
        desc = f" - {v.description}" if v.description else ""
        lines.append(f"* {link} — {v.published}{desc}")
    return (yaml_dump_frontmatter(meta)
            + f"\n# {name}\n\nChaîne YouTube, {len(matched)} vidéo(s).\n\n"
            + f"# Vidéos ({len(matched)})\n" + "\n".join(lines) + "\n")


def render_index(videos, entities, channels) -> str:
    idx = SRC_DIR / "index.md"
    out = [f'---\nokf_version: "{OKF_VERSION}"\n---\n', "# Entities",
           *[f"* {mdlink(n, idx, ent_dir() / f'{slugify(n)}.md')} - "
             f"{len([v for v in videos if n in v.entities])} vidéos"
             for n in sorted(entities, key=lambda n: -len([v for v in videos if n in v.entities]))],
           "\n# Channels",
           *[f"* {mdlink(n, idx, chan_dir() / f'{slugify(n)}.md')} - "
             f"{len([v for v in videos if v.channel == n])} vidéos" for n in sorted(channels)],
           "\n# Videos"]
    for v in sorted(videos, key=lambda v: v.published, reverse=True):
        desc = f" - {v.description}" if v.description else ""
        out.append(f"* {mdlink(v.title, idx, v.src)} — {v.channel}, {v.published}{desc}")
    return "\n".join(out) + "\n"


def render_subindex(kind: str, names, videos) -> str:
    base = ent_dir() if kind == "entities" else chan_dir()
    idx = base / "index.md"
    key = (lambda n: len([v for v in videos if n in v.entities])) if kind == "entities" \
        else (lambda n: len([v for v in videos if v.channel == n]))
    out = [f"# {kind.capitalize()}"]
    for n in sorted(names, key=lambda n: -key(n)):
        out.append(f"* {mdlink(n, idx, base / f'{slugify(n)}.md')} - {key(n)} vidéos")
    return "\n".join(out) + "\n"


# ── Build ─────────────────────────────────────────────────────────────
def collect_videos() -> list[Video]:
    files = sorted(p for p in SRC_DIR.rglob("*.md")
                   if "attachments" not in p.parts and not p.name.startswith(".")
                   and p.parent not in (ent_dir(), chan_dir())
                   and p.name not in ("index.md", "log.md"))
    return [Video(f) for f in files]


def build(dry_run: bool) -> None:
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    videos = collect_videos()
    entities = sorted({e for v in videos for e in v.entities})
    channels = sorted({v.channel for v in videos})
    print(f"Bundle : {SRC_DIR}")
    print(f"Vidéos : {len(videos)} | Entités : {len(entities)} | Chaînes : {len(channels)}\n")

    if dry_run:
        v = videos[0]
        print("=== DRY-RUN (aucune écriture) ===\n")
        print(f"── {v.src.relative_to(VAULT)}\n")
        txt = enriched_text(v)
        print("\n".join(txt.splitlines()[:12]) + "\n[... corps ...]\n"
              + "\n".join(txt.splitlines()[-6:]))
        sample = max(entities, key=lambda n: len([x for x in videos if n in x.entities]))
        print(f"\n── _entities/{slugify(sample)}.md\n")
        print("\n".join(render_entity(sample, videos, now).splitlines()[:12]))
        print("\n=== fin dry-run ===")
        return

    for v in videos:
        v.src.write_text(enriched_text(v), encoding="utf-8")
    ent_dir().mkdir(parents=True, exist_ok=True)
    chan_dir().mkdir(parents=True, exist_ok=True)
    for e in entities:
        (ent_dir() / f"{slugify(e)}.md").write_text(render_entity(e, videos, now), encoding="utf-8")
    for c in channels:
        (chan_dir() / f"{slugify(c)}.md").write_text(render_channel(c, videos, now), encoding="utf-8")
    (ent_dir() / "index.md").write_text(render_subindex("entities", entities, videos), encoding="utf-8")
    (chan_dir() / "index.md").write_text(render_subindex("channels", channels, videos), encoding="utf-8")
    (SRC_DIR / "index.md").write_text(render_index(videos, entities, channels), encoding="utf-8")
    (SRC_DIR / "log.md").write_text(
        f"# Update Log\n\n## {now}\n* **Build**: enrichi {len(videos)} vidéos, "
        f"{len(entities)} entités, {len(channels)} chaînes (OKF v{OKF_VERSION}).\n",
        encoding="utf-8")
    print(f"✅ Bundle OKF v{OKF_VERSION} : {len(videos)} vidéos enrichies + "
          f"{len(entities)} entités + {len(channels)} chaînes dans {SRC_DIR}")


def lint() -> None:
    """Resolve every relative markdown link in the bundle."""
    md = [p for p in SRC_DIR.rglob("*.md") if "attachments" not in p.parts]
    broken = 0
    for f in md:
        # match [text](dest); the leading (!?) lets us SKIP ![alt](img) image embeds
        for bang, bracketed, plain in re.findall(
                r"(!?)\[[^\]]*\]\((?:<([^>]*)>|([^)]*))\)", f.read_text(encoding="utf-8")):
            if bang == "!":
                continue
            dest = bracketed or plain
            if not dest or "://" in dest:   # skip external URLs (any scheme)
                continue
            # inside <> the '#' is a literal path char (CommonMark); only a plain
            # destination uses '#' as an anchor separator.
            path = bracketed if bracketed else plain.split("#", 1)[0]
            target = (f.parent / path).resolve()
            if not target.exists():
                print(f"CASSÉ  {f.relative_to(SRC_DIR)} → {dest}")
                broken += 1
    print(f"\nLiens markdown cassés : {broken}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build an OKF v0.1 bundle over YouTube summaries.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lint", action="store_true")
    ap.add_argument("--src", type=Path)
    args = ap.parse_args()
    global SRC_DIR
    if args.src:
        SRC_DIR = args.src
    if not SRC_DIR.is_dir():
        print(f"Bundle introuvable : {SRC_DIR}", file=sys.stderr)
        return 1
    if args.lint:
        lint()
        return 0
    build(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
