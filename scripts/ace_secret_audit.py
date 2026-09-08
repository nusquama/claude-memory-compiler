"""Read-only audit of potential secrets already stored in the ACE vault."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from config import VAULT_ROOT
from utils import sensitive_text_findings


def classify_path(project: Path, path: Path) -> str:
    rel = path.relative_to(project)
    if rel.parts[0] == "daily":
        return "daily"
    if rel.parts[0] == "knowledge":
        return "knowledge"
    return "state_log"


def audit_vault(vault: Path) -> dict[str, object]:
    totals: Counter[str] = Counter()
    by_project: dict[str, dict[str, object]] = {}
    flagged_files = 0

    projects = [
        path for path in vault.iterdir()
        if path.is_dir() and path.name != "_config" and not path.name.startswith(".")
    ]
    for project in sorted(projects):
        project_categories: Counter[str] = Counter()
        project_kinds: Counter[str] = Counter()
        project_files = 0
        candidates = list((project / "daily").glob("*.md"))
        candidates.extend((project / "knowledge").glob("**/*.md"))
        candidates.extend((project / ".state").glob("*.log"))
        for path in sorted(set(candidates)):
            try:
                findings = sensitive_text_findings(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
            if not findings:
                continue
            flagged_files += 1
            project_files += 1
            project_kinds[classify_path(project, path)] += 1
            project_categories.update(findings)
            totals.update(findings)
        if project_files:
            by_project[project.name] = {
                "flagged_files": project_files,
                "file_types": dict(sorted(project_kinds.items())),
                "categories": dict(sorted(project_categories.items())),
            }

    return {
        "status": "attention" if flagged_files else "ok",
        "flagged_files": flagged_files,
        "potential_values": sum(totals.values()),
        "categories": dict(sorted(totals.items())),
        "projects": by_project,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ACE vault secrets without showing values")
    parser.add_argument("--vault", default=str(VAULT_ROOT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_vault(Path(args.vault).expanduser())
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"ACE secret audit: {report['status']}")
        print(f"  flagged_files: {report['flagged_files']}")
        print(f"  potential_values: {report['potential_values']}")
        for project, details in report["projects"].items():
            print(
                f"  {project}: files={details['flagged_files']} "
                f"types={details['file_types']} categories={details['categories']}"
            )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
