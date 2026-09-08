"""Publish the local knowledge master and materialize immutable reader copies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from ace_database import MAX_COMPILED_BYTES, SupabaseStore
from ace_projects import resolve_project
from utils import redact_sensitive_text


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def checksum(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


_MARKDOWN_LINK_RE = re.compile(r"(?<!\!)\]\(\s*([^\s)]+)(?:\s+[^)]*)?\)")
_LEGACY_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]")
_BUNDLE_DIRS = ("concepts/", "connections/", "qa/", "daily/")


def _resolve_bundle_link(source: str, raw_target: str) -> tuple[str | None, str | None]:
    """Resolve one local Markdown target without touching the filesystem."""
    target = raw_target.strip().strip("<>").split("#", 1)[0].strip()
    if not target or target.startswith(("//", "#")):
        return None, None
    # External URLs and mail links are not files in a published bundle.
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
        return None, None

    root_relative = target.startswith("/")
    value = target.lstrip("/")
    # ACE articles use root-relative links (or the generated index's
    # ``concepts/...`` form).  Ignore prose/code examples such as ``path`` or
    # ``attachments/foo.png`` rather than treating them as bundle references.
    if not root_relative and not value.startswith(_BUNDLE_DIRS) and not value.startswith("../"):
        return None, None
    source_parts = [] if root_relative else list(PurePosixPath(source).parent.parts)
    parts = source_parts
    for part in PurePosixPath(value).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                return None, f"link escapes knowledge root: {raw_target}"
            parts.pop()
            continue
        parts.append(part)
    if not parts:
        return None, None
    resolved = PurePosixPath(*parts)
    # A link to a non-Markdown asset is external to the compiled corpus.  A
    # bare local article slug remains supported for backwards compatibility.
    if resolved.suffix and resolved.suffix.lower() != ".md":
        return None, None
    if not resolved.suffix:
        resolved = resolved.with_suffix(".md")
    return resolved.as_posix(), None


def _bundle_link_targets(source: str, content: str) -> list[tuple[str | None, str | None]]:
    """Extract local Markdown/legacy article links and their safe targets."""
    # Fenced examples document link syntax but do not constitute links in the
    # published reader graph.
    visible_lines: list[str] = []
    fenced = False
    for line in content.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if not fenced:
            visible_lines.append(line)
    visible = "\n".join(visible_lines)
    raw_targets = [match.group(1) for match in _MARKDOWN_LINK_RE.finditer(visible)]
    raw_targets.extend(
        match.group(1)
        for match in _LEGACY_LINK_RE.finditer(visible)
        if match.group(1).lstrip("/").startswith(_BUNDLE_DIRS)
    )
    return [_resolve_bundle_link(source, target) for target in raw_targets]


def _validate_bundle_files(files: dict[str, str]) -> list[str]:
    """Validate the structural relationships needed by a reader snapshot."""
    errors: list[str] = []
    if "index.md" not in files:
        errors.append("knowledge index is missing")

    index_targets: set[str] = set()
    if "index.md" in files:
        for target, error in _bundle_link_targets("index.md", files["index.md"]):
            if error:
                errors.append(error)
            elif target:
                index_targets.add(target)

    durable = sorted(
        name for name in files
        if name.startswith(("concepts/", "connections/")) and name.endswith(".md")
    )
    for name in durable:
        if name not in index_targets:
            errors.append(f"article missing from index: {name}")
    if durable and "log.md" not in files:
        errors.append("knowledge build log is missing")

    for source, content in files.items():
        for target, error in _bundle_link_targets(source, content):
            if error:
                errors.append(f"{source}: {error}")
                continue
            if not target or target.startswith("daily/"):
                # Daily source links point to the project source tree and are
                # intentionally outside a compiled reader snapshot.
                continue
            if target not in files:
                errors.append(f"broken internal link: {source} -> {target}")
    return errors


def publish_project(project: Any, store: Any = None) -> dict[str, Any]:
    store = store or SupabaseStore()
    info = project.as_dict() if hasattr(project, "as_dict") else dict(project)
    project_id = str(uuid.UUID(str(info.get("id") or info.get("project_id"))))
    root = Path(info["vault_dir"]) / "knowledge"
    if not root.is_dir() or root.is_symlink():
        raise ValueError("knowledge master is unavailable")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*.md")):
        if path.is_symlink() or root.resolve() not in path.resolve().parents:
            raise ValueError("knowledge source must not escape its master")
        files[path.relative_to(root).as_posix()] = redact_sensitive_text(path.read_text(encoding="utf-8"))
    if not files:
        return {"published": False, "reason": "empty_knowledge"}
    validation_errors = _validate_bundle_files(files)
    if validation_errors:
        raise ValueError(
            "knowledge master is incomplete: " + "; ".join(validation_errors[:5])
        )
    snapshot = {"schema_version": 1, "project_id": project_id, "files": files}
    if len(_bytes(snapshot)) > MAX_COMPILED_BYTES:
        raise ValueError("compiled snapshot exceeds the transport limit")
    prior = store.read_compiled_snapshot(project_id)
    digest = checksum(snapshot)
    if prior and prior["checksum"] == digest:
        return {"published": False, "unchanged": True, "version": prior["version"], "checksum": digest}
    version = int(prior["version"]) + 1 if prior else 1
    receipt = store.publish_compiled_snapshot(project_id, version, snapshot)
    return {"published": True, "version": version, "checksum": digest, "files": len(files), "receipt": receipt}


def fetch_reader_copy(project_id: str, *, store: Any = None, version: int | None = None, cache_root: Path | None = None) -> dict[str, Any]:
    project_id = str(uuid.UUID(project_id))
    record = (store or SupabaseStore()).read_compiled_snapshot(project_id, version)
    if not record:
        raise ValueError("no published knowledge version")
    snapshot = record["snapshot"]
    if checksum(snapshot) != record["checksum"] or snapshot.get("project_id") != project_id:
        raise ValueError("compiled snapshot identity or checksum mismatch")
    files = snapshot.get("files")
    if not isinstance(files, dict) or len(_bytes(snapshot)) > MAX_COMPILED_BYTES:
        raise ValueError("invalid compiled snapshot")
    root = cache_root or Path.home() / ".agents/private/ace/knowledge"
    target = root / project_id / str(int(record["version"]))
    for name, content in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.suffix != ".md" or not isinstance(content, str):
            raise ValueError("invalid knowledge reader path")
    validation_errors = _validate_bundle_files(files)
    if validation_errors:
        raise ValueError(
            "compiled snapshot is incomplete: " + "; ".join(validation_errors[:5])
        )
    if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
        raise ValueError("reader cache must not use symlinks")
    if target.exists():
        existing = {p.relative_to(target).as_posix(): p.read_text(encoding="utf-8") for p in target.rglob("*.md") if p.is_file()}
        if existing != files:
            raise ValueError("reader copy changed; refusing to replace it")
        return {"version": record["version"], "path": str(target), "unchanged": True, "checksum": record["checksum"]}
    if any(parent.is_symlink() for parent in [root, *root.parents, target.parent]):
        raise ValueError("reader cache must not use symlinks")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = Path(tempfile.mkdtemp(prefix=".version-", dir=target.parent))
    try:
        for name, content in files.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.write_text(content, encoding="utf-8")
            path.chmod(0o400)
        for directory in sorted((p for p in temporary.rglob("*") if p.is_dir()), reverse=True):
            directory.chmod(0o500)
        temporary.chmod(0o500)
        os.rename(temporary, target)
    except Exception:
        # Preserve the incomplete staging directory for inspection. It never
        # becomes the requested version, and no reader consumes it.
        raise
    return {"version": record["version"], "path": str(target), "files": len(files), "checksum": record["checksum"], "read_only": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--cwd", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--project", required=True)
    fetch.add_argument("--version", type=int)
    args = parser.parse_args(argv)
    try:
        if args.action == "publish":
            project = resolve_project(args.cwd, strict=True)
            if not project:
                raise ValueError("project is not initialized")
            result = publish_project(project)
        else:
            result = fetch_reader_copy(args.project, version=args.version)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
