"""Project identity and registration for the ACE ingestion boundary.

The registry deliberately has a small contract:

* a project is usable only when ``<vault>/<name>/knowledge`` exists;
* resolving a project never creates or changes anything;
* registration/initialisation is the only operation that writes the central
  marker (``<vault>/.state/ace-projects.json``), and the marker is replaced
  atomically;
* the legacy, marker-less project folders remain readable.  Their id is the
  deterministic UUID5 of the canonical checkout root until a registration
  binds the root explicitly.

This module is intentionally stdlib-only so it can be used by hooks and by a
collector before the rest of the CMC/ACE migration has settled.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
REGISTRY_FILENAME = "ace-projects.json"
DEFAULT_VAULT_ROOT = Path(__file__).resolve().parents[2]

# CMC wrote routing evidence in per-project state files before ACE gained a
# central registry.  A knowledge directory alone is not such evidence: it is
# cheap to create and its basename is not a source identity.  These are the
# state files whose routing fields may be used for a read-only migration.
_LEGACY_PROOF_FILES = {
    "codex-backfill.json",
    "collection-state.json",
    "state.json",
    "last-flush.json",
    "checkpoint-cursor.json",
    "native-summaries.json",
}
_LEGACY_ROOT_KEYS = {"root", "project_root", "source_cwd", "cwd", "working_directory"}


class ProjectRegistryError(RuntimeError):
    """Base class for fail-closed project identity errors."""


class ProjectNotInitialized(ProjectRegistryError):
    """The source root has no matching initialised vault project."""


class AmbiguousProjectError(ProjectRegistryError):
    """A basename cannot safely identify one source root."""


def _safe_name(name: str) -> str:
    value = str(name).strip()
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ProjectRegistryError(f"invalid project name: {name!r}")
    if "/" in value or "\\" in value:
        raise ProjectRegistryError(f"invalid project name: {name!r}")
    return value


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def canonical_git_root(start: str | Path) -> Path:
    """Return the canonical checkout root, including a linked worktree.

    The normal and worktree cases use Git's own identity information.  A
    non-Git synthetic directory is returned as its resolved path; this keeps
    the explicit registry API useful for fixtures and for projects whose
    source is mounted without a ``.git`` directory.  Resolution remains
    opt-in because the corresponding vault folder must already be
    initialised.
    """

    start_path = _resolved(start)
    if not start_path.exists():
        raise ProjectRegistryError(f"source root does not exist: {start_path}")
    if start_path.is_file():
        start_path = start_path.parent

    try:
        top = subprocess.run(
            ["git", "-C", str(start_path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        common = subprocess.run(
            ["git", "-C", str(start_path), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return start_path

    if top.returncode != 0 or not top.stdout.strip():
        return start_path

    root = _resolved(top.stdout.strip())
    if common.returncode == 0 and common.stdout.strip():
        common_dir = Path(common.stdout.strip())
        if not common_dir.is_absolute():
            common_dir = (start_path / common_dir).resolve()
        # For a linked worktree --show-toplevel is the worktree.  The shared
        # .git directory is the stable project identity we want instead.
        if common_dir.name == ".git" and common_dir.parent != root:
            root = common_dir.parent
    return root


def deterministic_project_id(root: str | Path) -> str:
    """Return the stable UUID5 used by marker-less projects."""

    canonical = canonical_git_root(root)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, canonical.as_posix()))


@dataclass(frozen=True)
class ProjectInfo:
    """Public project identity used in an ingestion envelope."""

    id: str | None
    name: str
    root: Path | None
    vault_dir: Path
    resolved: bool = True

    @property
    def project_id(self) -> str | None:
        """Compatibility spelling for callers that avoid the ``id`` field."""

        return self.id

    @property
    def processable(self) -> bool:
        """Whether the identity is safe to use for collection."""

        return bool(self.resolved and self.id and self.root)

    def as_dict(self) -> dict[str, str | None]:
        return {
            "id": str(self.id) if self.id else None,
            "name": self.name,
            "root": str(self.root) if self.root else None,
            "vault_dir": str(self.vault_dir),
        }

    def __getitem__(self, key: str) -> str | None:
        return self.as_dict()[key]


def _record_info(record: dict[str, Any], vault_root: Path) -> ProjectInfo:
    try:
        project_id = str(uuid.UUID(str(record["id"])))
        name = _safe_name(str(record["name"]))
        raw_root_text = str(record["root"]).strip()
        if not raw_root_text:
            raise ProjectRegistryError("project marker has no source root")
        raw_root = _resolved(raw_root_text)
        # A registered checkout may have been removed after initialization.
        # Preserve its canonical marker path so one stale record cannot make
        # every other explicit project unreadable; it simply will not match a
        # current source during resolve().
        root = canonical_git_root(raw_root) if raw_root.exists() else raw_root
    except (KeyError, ValueError, TypeError, ProjectRegistryError) as exc:
        raise ProjectRegistryError("invalid project registry marker") from exc
    vault_dir = _resolved(record.get("vault_dir") or (vault_root / name))
    if vault_dir != _resolved(vault_root / name):
        raise ProjectRegistryError("project marker escapes the vault")
    return ProjectInfo(id=project_id, name=name, root=root, vault_dir=vault_dir)


class ProjectRegistry:
    """Read/modify the central ACE project registry.

    ``resolve`` and ``list_initialized`` perform no filesystem writes.  The
    only mutating method is ``register`` (``init`` is an explicit alias).
    """

    # A process-local guard catches two unmarked checkout roots with the same
    # basename.  A central marker is required for a durable binding; until
    # one exists, the second distinct root is ambiguous and is rejected.
    _legacy_seen: dict[tuple[str, str], set[Path]] = {}

    def __init__(self, vault_root: str | Path = DEFAULT_VAULT_ROOT) -> None:
        self.vault_root = _resolved(vault_root)
        self.state_dir = self.vault_root / ".state"
        self.marker_path = self.state_dir / REGISTRY_FILENAME

    def _read_records(self) -> list[ProjectInfo]:
        if not self.marker_path.exists():
            return []
        try:
            raw = json.loads(self.marker_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProjectRegistryError("cannot read ACE project marker") from exc
        if not isinstance(raw, dict) or raw.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ProjectRegistryError("unsupported ACE project marker")
        entries = raw.get("projects", [])
        if isinstance(entries, dict):
            entries = list(entries.values())
        if not isinstance(entries, list):
            raise ProjectRegistryError("invalid ACE project marker projects")
        records = [_record_info(entry, self.vault_root) for entry in entries if isinstance(entry, dict)]
        # A hand-edited or interrupted marker must never make two same-name
        # bindings look like one project.  Fail closed before callers build a
        # routing index from the records.
        seen_names: dict[str, ProjectInfo] = {}
        seen_roots: dict[Path, ProjectInfo] = {}
        seen_ids: dict[str, ProjectInfo] = {}
        for record in records:
            prior_name = seen_names.get(record.name)
            if prior_name is not None and prior_name.root != record.root:
                raise AmbiguousProjectError(
                    f"project basename {record.name!r} is bound to multiple roots"
                )
            prior_root = seen_roots.get(record.root)
            if prior_root is not None and prior_root.id != record.id:
                raise AmbiguousProjectError("source root has multiple project bindings")
            prior_id = seen_ids.get(str(record.id))
            if prior_id is not None and prior_id.root != record.root:
                raise ProjectRegistryError("project id is bound to multiple source roots")
            seen_names[record.name] = record
            seen_roots[record.root] = record
            seen_ids[str(record.id)] = record
        return records

    def _write_records(self, records: Iterable[ProjectInfo]) -> None:
        """Atomically replace the central marker after explicit registration."""

        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "projects": [item.as_dict() for item in records],
        }
        fd, temporary = tempfile.mkstemp(
            dir=str(self.state_dir), prefix=".ace-projects-", suffix=".tmp"
        )
        temporary_path = Path(temporary)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.marker_path)
            try:
                directory_fd = os.open(self.state_dir, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _candidate(self, name: str) -> Path | None:
        safe = _safe_name(name)
        directory = self.vault_root / safe
        if not directory.is_dir() or not (directory / "knowledge").is_dir():
            return None
        return directory.resolve()

    def _legacy_roots(self, vault_dir: Path) -> set[Path]:
        """Read old per-project state for an existing source-root binding.

        This is deliberately conservative and read-only.  It accepts only
        fields that historically carried a checkout/cwd identity and ignores
        transcript paths, hashes, and arbitrary string values.
        """

        roots: set[Path] = set()
        state_dir = vault_dir / ".state"
        if not state_dir.is_dir():
            return roots
        keys = {item.replace("-", "_").lower() for item in _LEGACY_ROOT_KEYS}

        def normalised_key(value: Any) -> str:
            return str(value).replace("-", "_").lower()

        def has_root_field(value: Any) -> bool:
            """Return whether a legacy state object carries routing proof.

            CMC state is not trusted merely because a project folder exists.
            A root-bearing state file is the compatibility proof that the old
            collector had actually associated this vault folder with a
            checkout.  The recursive walk intentionally ignores transcript
            paths, hashes, and message content.
            """

            if isinstance(value, Mapping):
                if any(
                    normalised_key(key) in keys
                    and isinstance(child, str)
                    and bool(child.strip())
                    for key, child in value.items()
                ):
                    return True
                return any(has_root_field(child) for child in value.values())
            if isinstance(value, list):
                return any(has_root_field(child) for child in value)
            return False

        def visit(value: Any, key: str = "") -> None:
            if isinstance(value, Mapping):
                for child_key, child in value.items():
                    visit(child, str(child_key))
                return
            if normalised_key(key) not in keys or not isinstance(value, str) or not value:
                return
            candidate = Path(value).expanduser()
            if not candidate.exists() or not candidate.is_dir():
                return
            try:
                roots.add(canonical_git_root(candidate))
            except ProjectRegistryError:
                return

        try:
            state_files = sorted(state_dir.glob("*.json"))
            for state_file in state_files:
                if state_file.name not in _LEGACY_PROOF_FILES:
                    continue
                try:
                    document = json.loads(state_file.read_text(encoding="utf-8"))
                    if has_root_field(document):
                        visit(document)
                except (OSError, UnicodeError, json.JSONDecodeError):
                    continue
        except OSError:
            return roots
        return roots

    def register(
        self,
        root: str | Path,
        *,
        name: str | None = None,
        project_id: str | None = None,
        create: bool = True,
    ) -> ProjectInfo:
        """Explicitly initialise and bind a source root to a vault project."""

        canonical = canonical_git_root(root)
        project_name = _safe_name(name or canonical.name)
        vault_dir = (self.vault_root / project_name).resolve()
        if self.vault_root not in vault_dir.parents:
            raise ProjectRegistryError("project directory escapes vault")
        existing = self._read_records()

        for info in existing:
            if info.root == canonical:
                if info.name != project_name:
                    raise ProjectRegistryError("source root already has another project name")
                if project_id and str(uuid.UUID(str(project_id))) != info.id:
                    raise ProjectRegistryError("source root already has another project id")
                return info
            if info.name == project_name and info.root != canonical:
                raise AmbiguousProjectError(
                    f"project basename {project_name!r} is already bound to another root"
                )

        if vault_dir.exists() and not (vault_dir / "knowledge").is_dir():
            raise ProjectRegistryError(f"vault project is not initialised: {vault_dir}")
        if not create and not vault_dir.is_dir():
            raise ProjectNotInitialized(str(vault_dir))
        if create:
            # This write is intentionally reachable only through explicit
            # register/init, never through collection resolution.
            (vault_dir / "knowledge").mkdir(parents=True, exist_ok=True)

        chosen_id = str(uuid.UUID(str(project_id))) if project_id else deterministic_project_id(canonical)
        if any(info.id == chosen_id and info.root != canonical for info in existing):
            raise ProjectRegistryError("project id is already bound to another root")
        info = ProjectInfo(chosen_id, project_name, canonical, vault_dir)
        self._write_records([*existing, info])
        return info

    init = register

    def resolve(self, root: str | Path, *, strict: bool = True) -> ProjectInfo | None:
        """Resolve a source root without creating state or using a fallback."""

        canonical = canonical_git_root(root)
        project_name = _safe_name(canonical.name)
        records = self._read_records()
        # Name is not the identity once a project was explicitly registered;
        # resolve the authoritative root binding before looking at basename.
        same_root = [item for item in records if item.root == canonical]
        if len(same_root) == 1:
            info = same_root[0]
            if self._candidate(info.name) is None:
                return None
            return info
        if len(same_root) > 1:
            raise AmbiguousProjectError("source root has multiple project bindings")

        vault_dir = self._candidate(project_name)
        if vault_dir is None:
            return None

        same_name = [item for item in records if item.name == project_name]
        if same_name:
            raise AmbiguousProjectError(
                f"project basename {project_name!r} is bound to another root"
            )

        # Legacy projects have no marker yet.  Only an old CMC state file that
        # names this exact canonical root is authoritative.  A knowledge
        # folder (including one named ``Conversations``) is not permission.
        evidence = self._legacy_roots(vault_dir)
        if evidence and canonical not in evidence:
            raise AmbiguousProjectError(
                f"legacy project {project_name!r} is bound to another source root"
            )
        if canonical not in evidence:
            if strict:
                return None
            # Non-strict is retained as an explicit migration escape hatch for
            # callers that have already independently authenticated the source.
            # Production collection always passes strict=True.
            roots = self._legacy_seen.setdefault((str(self.vault_root), project_name), set())
            roots.add(canonical)
            if len(roots) > 1:
                raise AmbiguousProjectError(
                    f"unmarked project basename {project_name!r} is ambiguous; register it explicitly"
                )
        return ProjectInfo(
            deterministic_project_id(canonical), project_name, canonical, vault_dir
        )

    def list_initialized(self, *, include_legacy: bool = True) -> list[ProjectInfo]:
        """List knowledge folders without writing state.

        Marker-less folders are returned as ``processable=False`` records with
        no fabricated root/id.  Callers must resolve an observed source root
        or explicitly register it before collection.
        """

        records = self._read_records()
        by_name = {item.name: item for item in records}
        result: list[ProjectInfo] = []
        if not self.vault_root.is_dir():
            return result
        for directory in sorted(self.vault_root.iterdir(), key=lambda item: item.name):
            if not directory.is_dir() or directory.name == ".state":
                continue
            if not (directory / "knowledge").is_dir():
                continue
            marked = by_name.get(directory.name)
            if marked is not None:
                result.append(marked)
            elif include_legacy:
                result.append(
                    ProjectInfo(
                        id=None,
                        name=directory.name,
                        root=None,
                        vault_dir=directory.resolve(),
                        resolved=False,
                    )
                )
        return result


def init_project(
    root: str | Path,
    *,
    vault_root: str | Path = DEFAULT_VAULT_ROOT,
    name: str | None = None,
    project_id: str | None = None,
) -> ProjectInfo:
    """Explicit registration convenience function."""

    return ProjectRegistry(vault_root).register(
        root, name=name, project_id=project_id, create=True
    )


register_project = init_project


def resolve_project(
    root: str | Path,
    *,
    vault_root: str | Path = DEFAULT_VAULT_ROOT,
    strict: bool = True,
) -> ProjectInfo | None:
    """Read-only source-root resolver; it never falls back to another project."""

    return ProjectRegistry(vault_root).resolve(root, strict=strict)


resolve_initialized_project = resolve_project


def list_initialized_projects(
    *, vault_root: str | Path = DEFAULT_VAULT_ROOT
) -> list[ProjectInfo]:
    return ProjectRegistry(vault_root).list_initialized()


def _cli() -> int:
    parser = argparse.ArgumentParser(description="ACE project registry")
    parser.add_argument("--vault-root", default=str(DEFAULT_VAULT_ROOT))
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="explicitly initialise a project")
    init_parser.add_argument("root")
    init_parser.add_argument("--name")
    init_parser.add_argument("--id", dest="project_id")

    resolve_parser = subparsers.add_parser("resolve", help="read-only project resolution")
    resolve_parser.add_argument("root")
    list_parser = subparsers.add_parser("list", help="list initialised projects")
    del list_parser
    args = parser.parse_args()
    registry = ProjectRegistry(args.vault_root)
    if args.command == "init":
        result = registry.register(args.root, name=args.name, project_id=args.project_id)
        print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "resolve":
        result = registry.resolve(args.root)
        print(json.dumps(result.as_dict() if result else None, ensure_ascii=False, sort_keys=True))
        return 0 if result else 1
    if args.command == "list":
        print(json.dumps([item.as_dict() for item in registry.list_initialized()], ensure_ascii=False))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through --help/CLI smoke tests
    raise SystemExit(_cli())
