#!/usr/bin/env python3
"""Plan and apply the non-destructive CMC to ACE state migration.

The migration copies durable state and private reports into the ACE-named
locations.  It never removes the historical CMC locations.  A dry-run is the
default.  ``--apply`` is required before any destination file is created.

The files are copied as opaque bytes.  This preserves unknown state fields,
incident decisions, checkpoints, and historical report content exactly.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


HOME = Path.home()
# CMC remains the historical source.  ACE's live private root is shared by
# state and reports; keeping the destination here aligned prevents a second
# active state tree under ``~/.codex/ace``.
DEFAULT_STATE_SOURCE = HOME / ".codex" / "cmc"
DEFAULT_STATE_DESTINATION = HOME / ".agents" / "private" / "ace"
DEFAULT_PRIVATE_REPORT_SOURCE = HOME / ".agents" / "private" / "cmc"
DEFAULT_PRIVATE_REPORT_DESTINATION = HOME / ".agents" / "private" / "ace"
DEFAULT_DAILY_REPORT_SOURCE = HOME / ".codex" / "reports" / "cmc-daily"
DEFAULT_DAILY_REPORT_DESTINATION = DEFAULT_PRIVATE_REPORT_DESTINATION / "daily"
DEFAULT_EVALUATION_SOURCE = HOME / ".codex" / "reports" / "cmc-improvement" / "evaluations"
DEFAULT_EVALUATION_DESTINATION = DEFAULT_PRIVATE_REPORT_DESTINATION / "evaluations"
DEFAULT_OVERENGINEERING_SOURCE = HOME / ".codex" / "reports" / "overengineering"
DEFAULT_OVERENGINEERING_DESTINATION = DEFAULT_PRIVATE_REPORT_DESTINATION / "overengineering"

# Locks belong to a running process, not to durable state.  Copying one could
# make the new runtime appear busy after a migration.
EXCLUDED_STATE_NAMES = frozenset({
    "collection-state.lock",
    "overengineering-state.lock",
    "flush.lock",
})

# Keep an explicit list for the known global state contract.  The migration
# also copies other regular files below the old state root, except locks, so a
# later state addition is not silently lost.
KNOWN_STATE_NAMES = (
    "backfill.log",
    "checkpoint-state.json",
    "claude-stop-throttle.json",
    "collection-state.json",
    "computer-use-notify.log",
    "incident-tracking.json",
    "launcher.log",
    "overengineering-state.json",
)


@dataclass(frozen=True)
class MigrationItem:
    """One opaque file considered by the migration plan."""

    category: str
    relative: str
    source: Path
    destination: Path
    status: str

    def as_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "relative": self.relative,
            "source": str(self.source),
            "destination": str(self.destination),
            "status": self.status,
            "action": self.status,
        }


def _is_regular_file(path: Path) -> bool:
    """Return true only for regular files, without following symlinks."""

    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def _relative_files(root: Path, *, include_known_missing: bool = False) -> list[Path]:
    """List safe regular files below ``root`` in deterministic order."""

    if not root.is_dir() or root.is_symlink():
        return []

    found: set[Path] = set()
    # ``rglob`` does not follow directory symlinks on the supported hosts, but
    # filter every result explicitly because this path controls file writes.
    for candidate in root.rglob("*"):
        if _is_regular_file(candidate) and candidate.name not in EXCLUDED_STATE_NAMES:
            found.add(candidate)

    # Keep this explicit list visible in the implementation and ensure a
    # future caller can audit the contract without inspecting current state.
    if include_known_missing:
        for name in KNOWN_STATE_NAMES:
            candidate = root / name
            if _is_regular_file(candidate):
                found.add(candidate)
    return sorted(found, key=lambda item: str(item.relative_to(root)))


def _state_files(root: Path) -> list[Path]:
    return _relative_files(root, include_known_missing=True)


def _opaque_same(source: Path, destination: Path) -> bool:
    """Compare file bytes without parsing or printing state content."""

    try:
        return filecmp.cmp(source, destination, shallow=False)
    except OSError:
        return False


def _plan_tree(
    category: str,
    source_root: Path,
    destination_root: Path,
    source_files: Iterable[Path],
    archive_root: Path | None = None,
) -> list[MigrationItem]:
    items: list[MigrationItem] = []
    for source in source_files:
        try:
            relative_path = source.relative_to(source_root)
        except ValueError:
            continue
        destination = destination_root / relative_path
        if destination.exists() or destination.is_symlink():
            status = "identical" if _opaque_same(source, destination) else "collision"
        else:
            status = "copy"
        if status == "collision" and archive_root is not None:
            archive_destination = archive_root / relative_path
            if archive_destination.exists() or archive_destination.is_symlink():
                status = "identical" if _opaque_same(source, archive_destination) else "collision"
                destination = archive_destination
            else:
                status = "archive"
                destination = archive_destination
        items.append(
            MigrationItem(
                category=category,
                relative=relative_path.as_posix(),
                source=source,
                destination=destination,
                status=status,
            )
        )
    return items


def build_plan(
    *,
    state_source: Path = DEFAULT_STATE_SOURCE,
    state_destination: Path = DEFAULT_STATE_DESTINATION,
    private_report_source: Path = DEFAULT_PRIVATE_REPORT_SOURCE,
    private_report_destination: Path = DEFAULT_PRIVATE_REPORT_DESTINATION,
    daily_report_source: Path = DEFAULT_DAILY_REPORT_SOURCE,
    daily_report_destination: Path = DEFAULT_DAILY_REPORT_DESTINATION,
    evaluation_source: Path = DEFAULT_EVALUATION_SOURCE,
    evaluation_destination: Path = DEFAULT_EVALUATION_DESTINATION,
    overengineering_source: Path = DEFAULT_OVERENGINEERING_SOURCE,
    overengineering_destination: Path = DEFAULT_OVERENGINEERING_DESTINATION,
) -> list[MigrationItem]:
    """Build a deterministic plan without writing or changing source state."""

    # The deployed ACE state directory may already contain a newer native
    # state file.  Preserve the legacy file under an archive in that one
    # default migration; synthetic/custom roots keep strict collision errors
    # so callers can decide explicitly.
    state_archive = (
        state_destination / "archive" / "cmc"
        if state_source == DEFAULT_STATE_SOURCE and state_destination == DEFAULT_STATE_DESTINATION
        else None
    )
    roots = (
        ("state", state_source, state_destination, _state_files(state_source), state_archive),
        (
            "private-report",
            private_report_source,
            private_report_destination,
            _relative_files(private_report_source),
            private_report_destination / "archive" / "cmc",
        ),
        (
            "daily-report",
            daily_report_source,
            daily_report_destination,
            _relative_files(daily_report_source),
            private_report_destination / "archive" / "cmc-daily",
        ),
        (
            "evaluation-report",
            evaluation_source,
            evaluation_destination,
            _relative_files(evaluation_source),
            private_report_destination / "archive" / "cmc-improvement",
        ),
        (
            "overengineering-report",
            overengineering_source,
            overengineering_destination,
            _relative_files(overengineering_source),
            private_report_destination / "archive" / "overengineering",
        ),
    )
    plan: list[MigrationItem] = []
    for category, source_root, destination_root, source_files, archive_root in roots:
        plan.extend(_plan_tree(category, source_root, destination_root, source_files, archive_root))
    return plan


# Public names kept explicit for callers embedding the migration without
# invoking the command-line interface.
plan_migration = build_plan


def _copy_without_overwrite(item: MigrationItem) -> None:
    """Copy one file atomically and fail closed if a destination appears."""

    destination = item.destination
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(str(destination))

    source_mode = stat.S_IMODE(item.source.stat().st_mode)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.ace-migrate-",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, source_mode)
        with item.source.open("rb") as source_handle, os.fdopen(fd, "wb") as destination_handle:
            fd = -1
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        # A hard-link install gives the destination an exclusive create step.
        # It avoids replacing a file created by another process after planning.
        os.link(temporary, destination)
        temporary.unlink(missing_ok=True)
    finally:
        if fd != -1:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def apply_plan(plan: Sequence[MigrationItem]) -> dict[str, int]:
    """Apply only ``copy`` items and leave all sources untouched."""

    copied = 0
    collisions = 0
    failed = 0
    for item in plan:
        if item.status == "identical":
            continue
        if item.status == "collision":
            collisions += 1
            continue
        if item.status not in {"copy", "archive"}:
            continue
        try:
            _copy_without_overwrite(item)
        except FileExistsError:
            collisions += 1
        except (OSError, shutil.Error):
            failed += 1
        else:
            copied += 1
    return {"copied": copied, "collisions": collisions, "failed": failed}


def migrate(*, apply: bool = False, **roots: Path) -> dict[str, object]:
    """Return a migration result, applying only when ``apply`` is true."""

    plan = build_plan(**roots)
    result: dict[str, object] = summarize_plan(plan)
    result["mode"] = "apply" if apply else "dry-run"
    result.update(apply_plan(plan) if apply else {"copied": 0, "failed": 0})
    return result


def summarize_plan(plan: Sequence[MigrationItem]) -> dict[str, int]:
    summary = {"copy": 0, "archive": 0, "identical": 0, "collision": 0}
    for item in plan:
        summary[item.status] = summary.get(item.status, 0) + 1
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Create missing ACE copies")
    parser.add_argument("--json", action="store_true", help="Print the plan as JSON")
    parser.add_argument("--state-source", default=str(DEFAULT_STATE_SOURCE))
    parser.add_argument("--state-destination", default=str(DEFAULT_STATE_DESTINATION))
    parser.add_argument("--private-report-source", default=str(DEFAULT_PRIVATE_REPORT_SOURCE))
    parser.add_argument(
        "--private-report-destination", default=str(DEFAULT_PRIVATE_REPORT_DESTINATION)
    )
    parser.add_argument("--daily-report-source", default=str(DEFAULT_DAILY_REPORT_SOURCE))
    parser.add_argument(
        "--daily-report-destination", default=str(DEFAULT_DAILY_REPORT_DESTINATION)
    )
    parser.add_argument("--evaluation-source", default=str(DEFAULT_EVALUATION_SOURCE))
    parser.add_argument("--evaluation-destination", default=str(DEFAULT_EVALUATION_DESTINATION))
    parser.add_argument(
        "--overengineering-source", default=str(DEFAULT_OVERENGINEERING_SOURCE)
    )
    parser.add_argument(
        "--overengineering-destination", default=str(DEFAULT_OVERENGINEERING_DESTINATION)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_plan(
        state_source=Path(args.state_source).expanduser(),
        state_destination=Path(args.state_destination).expanduser(),
        private_report_source=Path(args.private_report_source).expanduser(),
        private_report_destination=Path(args.private_report_destination).expanduser(),
        daily_report_source=Path(args.daily_report_source).expanduser(),
        daily_report_destination=Path(args.daily_report_destination).expanduser(),
        evaluation_source=Path(args.evaluation_source).expanduser(),
        evaluation_destination=Path(args.evaluation_destination).expanduser(),
        overengineering_source=Path(args.overengineering_source).expanduser(),
        overengineering_destination=Path(args.overengineering_destination).expanduser(),
    )
    before = summarize_plan(plan)
    result = dict(before)
    result["mode"] = "apply" if args.apply else "dry-run"
    if args.apply:
        result.update(apply_plan(plan))
    else:
        result.update({"copied": 0, "failed": 0})

    if args.json:
        print(json.dumps({"items": [item.as_dict() for item in plan], "summary": result}, indent=2))
    else:
        print(
            f"ACE migration {result['mode']}: copy={result['copy']} "
            f"archive={result['archive']} identical={result['identical']} "
            f"collision={result['collision']} "
            f"copied={result['copied']} failed={result['failed']}"
        )
        for item in plan:
            print(f"{item.status}\t{item.category}\t{item.relative}")

    return 2 if result["collision"] or result.get("collisions", 0) or result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
