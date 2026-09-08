#!/usr/bin/env python3
"""Safe native service manager for the ACE scheduler.

``ace tick`` owns persistent state, catch-up after sleep, daily/weekly
decisions, and the single-processor lock.  Central-host Linux scheduling is
collector-only and runs ``ace collect --sync``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform as host_platform
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

RUNTIME_ROOT = Path("/Users/franck/Library/CloudStorage/Dropbox/Claude_code/_config")
ACE_BIN = Path("/Users/franck/.agents/bin/ace")
LABEL, INTERVAL_SECONDS = "com.agentcentral.ace", 1800
TIMEZONE, DAILY_START, REPORT_TARGET, WEEKLY_DAY = "Europe/Paris", "07:00", "08:00", "Sunday"
SERVICE_NAME, TIMER_NAME = "agentcentral-ace-collect.service", "agentcentral-ace-collect.timer"
LAUNCHCTL, SYSTEMCTL = Path("/bin/launchctl"), Path("/usr/bin/systemctl")


class SchedulerError(RuntimeError):
    """A configuration, ownership, or service-manager failure."""


def _platform_name(value: str | None = None) -> str:
    value = (value or host_platform.system()).lower()
    if value in {"darwin", "mac", "macos"}:
        return "macos"
    if value in {"linux", "gnu/linux"}:
        return "linux"
    raise SchedulerError(f"unsupported scheduler platform: {value}")


def _absolute(path: Path | str) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise SchedulerError(f"path must be absolute: {path}")
    return path.absolute()


def resolve_runtime_python(runtime_root: Path, explicit: Path | str | None = None) -> Path:
    if explicit:
        return _absolute(explicit)
    for candidate in (runtime_root / ".venv/bin/python3", runtime_root / ".venv/bin/python"):
        if candidate.exists():
            # Keep the virtualenv entry point: resolving its symlink loses
            # the pyvenv.cfg context and the installed runtime dependencies.
            return candidate.absolute()
    return _absolute(sys.executable)


@dataclass(frozen=True)
class SchedulerConfig:
    runtime_root: Path = RUNTIME_ROOT
    home: Path = field(default_factory=Path.home)
    ace_bin: Path = ACE_BIN
    platform_name: str = field(default_factory=lambda: _platform_name())
    python_bin: Path | None = None
    remote: bool = False
    central_host: bool = False

    def __post_init__(self) -> None:
        root, home, ace = _absolute(self.runtime_root), _absolute(self.home), _absolute(self.ace_bin)
        object.__setattr__(self, "runtime_root", root)
        object.__setattr__(self, "home", home)
        object.__setattr__(self, "ace_bin", ace)
        object.__setattr__(self, "platform_name", _platform_name(self.platform_name))
        object.__setattr__(self, "python_bin", resolve_runtime_python(root, self.python_bin))

    @property
    def mode(self) -> str:
        return "collector-only" if self.platform_name == "linux" or self.remote else "tick"

    @property
    def private_dir(self) -> Path:
        return self.home / ".agents/private/ace"

    @property
    def launchd_path(self) -> Path:
        return self.home / "Library/LaunchAgents" / f"{LABEL}.plist"

    @property
    def systemd_dir(self) -> Path:
        return self.home / ".config/systemd/user"

    @property
    def state_path(self) -> Path:
        return self.private_dir / "scheduler-state.json"


def _config(config: SchedulerConfig | None = None, **overrides: Any) -> SchedulerConfig:
    return SchedulerConfig(**overrides) if config is None else replace(config, **overrides) if overrides else config


def scheduler_command(config: SchedulerConfig | None = None) -> list[str]:
    config = _config(config)
    return [str(config.ace_bin), "collect", "--sync"] if config.mode == "collector-only" else [str(config.ace_bin), "tick"]


def schedule_contract() -> dict[str, Any]:
    return {
        "tick_interval_seconds": INTERVAL_SECONDS,
        "timezone": TIMEZONE,
        "daily_start": DAILY_START,
        "daily_report_target": REPORT_TARGET,
        "weekly_day": WEEKLY_DAY,
        "weekly_report_target": REPORT_TARGET,
        "weekly_rule": f"{WEEKLY_DAY} {REPORT_TARGET} {TIMEZONE}",
        "catch_up": "owned by ace tick persistent state",
        "single_processor": "owned by ace tick lock on the configured Mac",
    }


def _environment(config: SchedulerConfig) -> dict[str, str]:
    entries = [config.ace_bin.parent, config.runtime_root / ".venv/bin", config.python_bin.parent,
               config.home / ".bun/bin",
               Path("/Users/franck/.local/bin"), Path("/opt/homebrew/bin"), Path("/usr/local/bin"),
               Path("/usr/bin"), Path("/bin")]
    path = list(dict.fromkeys(map(str, entries)))
    return {
        "HOME": str(config.home),
        "PATH": os.pathsep.join(path),
        "ACE_RUNTIME_DIR": str(config.runtime_root),
        "ACE_RUNTIME_PYTHON": str(config.python_bin),
        "ACE_AUTOMATION_MODE": "incremental",
        "ACE_SCHEDULE_TIMEZONE": TIMEZONE,
        "ACE_DAILY_START": DAILY_START,
        "ACE_DAILY_REPORT_TARGET": REPORT_TARGET,
        "ACE_WEEKLY_DAY": WEEKLY_DAY,
        "ACE_WEEKLY_REPORT_TARGET": REPORT_TARGET,
        "ACE_PRIVATE_DIR": str(config.private_dir),
    }


def build_launchd_plist(config: SchedulerConfig | None = None, **overrides: Any) -> bytes:
    config = _config(config, **overrides)
    data = {
        "Label": LABEL,
        "ProgramArguments": scheduler_command(config),
        "WorkingDirectory": str(config.runtime_root),
        "StartInterval": INTERVAL_SECONDS,
        "RunAtLoad": True,
        "KeepAlive": False,
        "ProcessType": "Background",
        "EnvironmentVariables": _environment(config),
        "StandardOutPath": str(config.private_dir / "launchd.log"),
        "StandardErrorPath": str(config.private_dir / "launchd.error.log"),
        "Comment": "StartInterval may be skipped during sleep; ace tick reconciles due work at wake.",
    }
    return plistlib.dumps(data, fmt=plistlib.FMT_XML, sort_keys=False)


def _unit_path(value: Path | str) -> str:
    return shlex.quote(str(value))


def build_systemd_service(config: SchedulerConfig | None = None, **overrides: Any) -> str:
    config = _config(config, **overrides)
    env = _environment(config)
    return "\n".join([
        "[Unit]", "Description=Agent Central ACE collector", "After=network-online.target", "",
        "[Service]", "Type=oneshot", f"ExecStart={_unit_path(config.ace_bin)} collect --sync",
        f"WorkingDirectory={_unit_path(config.runtime_root)}", f"Environment=PATH={env['PATH']}",
        f"Environment=ACE_RUNTIME_DIR={_unit_path(config.runtime_root)}",
        f"Environment=ACE_RUNTIME_PYTHON={_unit_path(config.python_bin)}",
        f"Environment=ACE_SCHEDULE_TIMEZONE={TIMEZONE}", f"Environment=ACE_PRIVATE_DIR={_unit_path(config.private_dir)}",
        "UMask=0077", f"StandardOutput=append:{_unit_path(config.private_dir / 'systemd.log')}",
        f"StandardError=append:{_unit_path(config.private_dir / 'systemd.error.log')}", "",
        "[Install]", "WantedBy=default.target", "",
    ])


def build_systemd_timer(config: SchedulerConfig | None = None, **overrides: Any) -> str:
    _config(config, **overrides)
    return "\n".join([
        "[Unit]", "Description=Persistent Agent Central ACE collector schedule", "",
        "[Timer]", "OnBootSec=30s", f"OnUnitActiveSec={INTERVAL_SECONDS}", "Persistent=true", "AccuracySec=1s",
        f"Unit={SERVICE_NAME}", "", "[Install]", "WantedBy=timers.target", "",
    ])


def build_systemd_units(config: SchedulerConfig | None = None, **overrides: Any) -> dict[str, str]:
    config = _config(config, **overrides)
    return {"service": build_systemd_service(config), "timer": build_systemd_timer(config)}


def managed_artifacts(config: SchedulerConfig | None = None) -> dict[Path, bytes]:
    config = _config(config)
    if config.platform_name == "macos":
        return {config.launchd_path: build_launchd_plist(config)}
    return {config.systemd_dir / SERVICE_NAME: build_systemd_service(config).encode(),
            config.systemd_dir / TIMER_NAME: build_systemd_timer(config).encode()}


def managed_paths(config: SchedulerConfig | None = None) -> list[Path]:
    return list(managed_artifacts(config))


def build_plan(config: SchedulerConfig | None = None) -> dict[str, Any]:
    config = _config(config)
    required = not config.central_host
    return {
        "platform": config.platform_name,
        "mode": config.mode,
        "service_manager": "launchd" if config.platform_name == "macos" else "systemd-user",
        "command": scheduler_command(config),
        "managed_files": [str(path) for path in managed_artifacts(config)],
        "state_file": str(config.state_path),
        "schedule": schedule_contract(),
        "central_host_opt_in_required": required,
        "warnings": ["Scheduler installation requires explicit --central-host."] if required else [],
        "dry_run": True,
    }


def _guard_parents(path: Path) -> None:
    for parent in path.parents:
        if parent.is_symlink() and parent not in {Path("/var"), Path("/tmp")}:
            raise SchedulerError(f"refusing symlink parent: {parent}")


def _ensure_dir(path: Path) -> None:
    _guard_parents(path)
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise SchedulerError(f"refusing unsafe directory: {path}")
    # Shared parents such as ~/Library/LaunchAgents are operator-owned.
    # Do not change their existing permissions when installing ACE.
    path.mkdir(parents=True, exist_ok=True, mode=0o700)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_state(config: SchedulerConfig) -> dict[str, Any] | None:
    path = config.state_path
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink():
        raise SchedulerError(f"refusing symlink state file: {path}")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SchedulerError(f"invalid scheduler state: {path}") from exc
    if not isinstance(state, dict) or not isinstance(state.get("managed_files"), list):
        raise SchedulerError(f"invalid scheduler state: {path}")
    return state


def _atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    _guard_parents(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temp, mode)
    os.replace(temp, path)


def _backup(path: Path) -> Path:
    for index in range(1000):
        backup = Path(f"{path}.bak" if index == 0 else f"{path}.bak.{index}")
        if backup.is_symlink():
            raise SchedulerError(f"refusing symlink backup: {backup}")
        if not backup.exists():
            shutil.copy2(path, backup)
            try:
                backup.chmod(0o600)
            except OSError:
                pass
            return backup
    raise SchedulerError(f"too many backups for {path}")


def _records(state: dict[str, Any] | None) -> dict[Path, dict[str, Any]]:
    records: dict[Path, dict[str, Any]] = {}
    for record in (state or {}).get("managed_files", []):
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            raise SchedulerError("invalid managed file record")
        records[_absolute(record["path"])] = record
    return records


def _validate_targets(artifacts: dict[Path, bytes], state: dict[str, Any] | None) -> None:
    owned = _records(state)
    for path in artifacts:
        _guard_parents(path)
        if path.is_symlink():
            raise SchedulerError(f"refusing symlink target: {path}")
        if path.exists() and (path.is_dir() or path not in owned):
            raise SchedulerError(f"target collision; file is not managed by ACE: {path}")


def _require_central_opt_in(config: SchedulerConfig) -> None:
    if not config.central_host:
        raise SchedulerError("scheduler installation is opt-in; pass --central-host")


def install(config: SchedulerConfig | None = None, *, apply: bool = False, activate: bool = False,
            runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    config = _config(config)
    artifacts = managed_artifacts(config)
    state = _read_state(config)
    _validate_targets(artifacts, state)
    result = build_plan(config) | {"operation": "install", "applied": False}
    if not apply:
        return result
    _require_central_opt_in(config)
    _ensure_dir(config.private_dir)
    for path in artifacts:
        _ensure_dir(path.parent)
    backups: list[str] = []
    if config.state_path.exists():
        backups.append(str(_backup(config.state_path)))
    for path in artifacts:
        if path.exists():
            backups.append(str(_backup(path)))
    for path, data in artifacts.items():
        _atomic_write(path, data)
    new_state = {"schema": 1, "label": LABEL, "platform": config.platform_name, "mode": config.mode,
                 "managed_files": [{"path": str(path), "sha256": _digest(path)} for path in artifacts],
                 "backups": backups, "disabled": False}
    _atomic_write(config.state_path, json.dumps(new_state, indent=2, sort_keys=True).encode() + b"\n")
    if activate:
        run_service_action(config, "start", runner=runner)
    return build_plan(config) | {"operation": "install", "applied": True, "dry_run": False, "backups": backups}


def disable(config: SchedulerConfig | None = None, *, apply: bool = False, activate: bool = False,
            runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    config = _config(config)
    state = _read_state(config)
    result = build_plan(config) | {"operation": "disable", "applied": False, "disabled": bool(state and state.get("disabled"))}
    if not apply or not state:
        return result
    if activate:
        run_service_action(config, "stop", runner=runner)
    state = dict(state)
    state["disabled"] = True
    _atomic_write(config.state_path, json.dumps(state, indent=2, sort_keys=True).encode() + b"\n")
    return result | {"applied": True, "disabled": True}


def uninstall(config: SchedulerConfig | None = None, *, apply: bool = False, activate: bool = False,
              runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    config = _config(config)
    state = _read_state(config)
    result = build_plan(config) | {"operation": "uninstall", "applied": False, "removed": []}
    if not apply or not state:
        return result
    records = _records(state)
    for path, record in records.items():
        _guard_parents(path)
        if path.is_symlink() or (path.exists() and (path.is_dir() or _digest(path) != str(record["sha256"]))):
            raise SchedulerError(f"refusing to remove changed managed file: {path}")
    if activate:
        run_service_action(config, "stop", runner=runner)
    removed: list[str] = []
    for path in records:
        if path.exists():
            path.unlink()
            removed.append(str(path))
    if config.state_path.exists():
        config.state_path.unlink()
        removed.append(str(config.state_path))
    return result | {"applied": True, "removed": removed}


def status(config: SchedulerConfig | None = None, *, probe: bool = False,
           runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    config = _config(config)
    state = _read_state(config)
    records = _records(state)
    files = [{"path": str(path), "status": "missing" if not path.exists() else
              "changed" if path.is_symlink() or _digest(path) != str(record["sha256"]) else "ok"}
             for path, record in records.items()]
    result = {"operation": "status", "platform": config.platform_name, "installed": bool(state),
              "disabled": bool(state and state.get("disabled")), "files": files,
              "state_file": str(config.state_path),
              "service_manager": "launchd" if config.platform_name == "macos" else "systemd-user"}
    if probe and state:
        try:
            run_service_action(config, "status", runner=runner)
        except SchedulerError as exc:
            raise SchedulerError(f"managed files exist but service status is unavailable: {exc}") from exc
        result["service_status"] = "active"
    return result


def service_command(config: SchedulerConfig, action: str) -> list[str]:
    if config.platform_name == "macos":
        target = f"gui/{os.getuid()}/{LABEL}"
        return {"start": [str(LAUNCHCTL), "bootstrap", f"gui/{os.getuid()}", str(config.launchd_path)],
                "stop": [str(LAUNCHCTL), "bootout", target], "status": [str(LAUNCHCTL), "print", target]}[action]
    return {"start": [str(SYSTEMCTL), "--user", "enable", "--now", TIMER_NAME],
            "stop": [str(SYSTEMCTL), "--user", "disable", "--now", TIMER_NAME],
            "status": [str(SYSTEMCTL), "--user", "is-active", TIMER_NAME]}[action]


def run_service_action(config: SchedulerConfig, action: str, *, runner: Callable[..., Any] = subprocess.run) -> Any:
    if config.platform_name == "linux" and action == "start":
        commands = [[str(SYSTEMCTL), "--user", "daemon-reload"], service_command(config, action)]
    elif config.platform_name == "macos" and action == "start":
        target = f"gui/{os.getuid()}/{LABEL}"
        commands = [[str(LAUNCHCTL), "enable", target], service_command(config, action)]
    else:
        commands = [service_command(config, action)]
    result = None
    for command in commands:
        result = runner(command, check=False, capture_output=True, text=True)
        if getattr(result, "returncode", 0) != 0:
            raise SchedulerError(f"service manager failed ({action}): {command[0]}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or safely manage the native ACE scheduler")
    parser.add_argument("action", nargs="?", choices=("plan", "install", "status", "disable", "uninstall"), default="plan")
    parser.add_argument("--platform", dest="platform_name", choices=("macos", "darwin", "linux"))
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--ace-bin", type=Path, default=ACE_BIN)
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--remote", action="store_true", help="collector-only mode")
    parser.add_argument("--central-host", action="store_true", help="explicitly opt into central scheduling")
    parser.add_argument("--apply", action="store_true", help="allow file/state changes")
    parser.add_argument("--dry-run", action="store_true", help="force a read-only plan")
    parser.add_argument("--activate", action="store_true", help="also call the service manager (requires --apply)")
    parser.add_argument("--json", action="store_true")
    return parser


def _print_result(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"{result.get('operation', 'plan')}: {'applied' if result.get('applied') else 'dry-run'}")
    if result.get("command"):
        print("command:", " ".join(shlex.quote(value) for value in result["command"]))
    for item in result.get("managed_files", result.get("files", [])):
        print(item if isinstance(item, str) else f"{item['status']}: {item['path']}")
    for warning in result.get("warnings", []):
        print("warning:", warning)


plan = build_plan
render_launchd_plist = build_launchd_plist
render_systemd_service = build_systemd_service
render_systemd_timer = build_systemd_timer


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = SchedulerConfig(runtime_root=args.runtime_root, home=args.home, ace_bin=args.ace_bin,
                                 platform_name=args.platform_name or host_platform.system(), python_bin=args.runtime_python,
                                 remote=args.remote, central_host=args.central_host)
        if args.dry_run and args.apply:
            raise SchedulerError("--dry-run cannot be combined with --apply")
        if args.activate and (not args.apply or args.dry_run):
            raise SchedulerError("--activate requires --apply")
        if args.action == "plan":
            result = build_plan(config)
        elif args.action == "install":
            result = install(config, apply=args.apply, activate=args.activate)
        elif args.action == "disable":
            result = disable(config, apply=args.apply, activate=args.activate)
        elif args.action == "uninstall":
            result = uninstall(config, apply=args.apply, activate=args.activate)
        else:
            result = status(config, probe=True)
        _print_result(result, args.json)
        return 0
    except (SchedulerError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
