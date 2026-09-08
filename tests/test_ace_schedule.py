from __future__ import annotations

import plistlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import ace_schedule as schedule


class AceScheduleTests(unittest.TestCase):
    def test_python_keeps_virtualenv_entry_point(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp) / "runtime"
            base = Path(tmp) / "python3"
            base.touch()
            candidate = runtime / ".venv/bin/python3"
            candidate.parent.mkdir(parents=True)
            candidate.symlink_to(base)
            self.assertEqual(schedule.resolve_runtime_python(runtime), candidate)
            self.assertEqual(schedule.resolve_runtime_python(runtime, candidate), candidate)

    def test_install_preserves_shared_parent_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            shared = config.launchd_path.parent
            shared.mkdir(parents=True)
            shared.chmod(0o755)
            result = schedule.install(config, apply=True)
            self.assertEqual(shared.stat().st_mode & 0o777, 0o755)
            self.assertFalse(result["dry_run"])

    def _config(self, root: Path, *, platform_name: str = "macos", central_host: bool = True) -> schedule.SchedulerConfig:
        runtime = root / "runtime"
        runtime.mkdir()
        python_bin = root / "python" / "bin" / "python3"
        python_bin.parent.mkdir(parents=True)
        python_bin.touch()
        return schedule.SchedulerConfig(
            runtime_root=runtime,
            home=root / "home",
            ace_bin=root / "agents" / "bin" / "ace",
            python_bin=python_bin,
            platform_name=platform_name,
            central_host=central_host,
        )

    def test_launchd_contract_is_native_tick_and_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            plist = plistlib.loads(schedule.build_launchd_plist(config))
            self.assertEqual(plist["Label"], schedule.LABEL)
            self.assertEqual(plist["StartInterval"], 1800)
            self.assertTrue(plist["RunAtLoad"])
            self.assertEqual(plist["ProgramArguments"], [str(config.ace_bin), "tick"])
            self.assertNotIn("model", repr(plist).lower())
            self.assertNotIn("conversation", repr(plist).lower())
            self.assertEqual(plist["WorkingDirectory"], str(config.runtime_root))
            environment = plist["EnvironmentVariables"]
            self.assertIn(str(config.ace_bin.parent), environment["PATH"])
            self.assertIn(str(config.python_bin), environment["ACE_RUNTIME_PYTHON"])
            self.assertEqual(environment["ACE_AUTOMATION_MODE"], "incremental")
            self.assertEqual(environment["ACE_SCHEDULE_TIMEZONE"], "Europe/Paris")
            self.assertEqual(environment["ACE_DAILY_START"], "07:00")
            self.assertEqual(environment["ACE_DAILY_REPORT_TARGET"], "08:00")

    def test_linux_units_are_persistent_collector_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp), platform_name="linux", central_host=True)
            service = schedule.build_systemd_service(config)
            timer = schedule.build_systemd_timer(config)
            self.assertIn(f"ExecStart={config.ace_bin} collect --sync", service)
            self.assertNotIn(" tick", service)
            self.assertIn("OnUnitActiveSec=1800", timer)
            self.assertIn("Persistent=true", timer)

    def test_default_plan_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            result = schedule.install(config)
            self.assertFalse(result["applied"])
            self.assertFalse(config.state_path.exists())
            self.assertFalse(config.launchd_path.exists())

    def test_install_rejects_unowned_collision_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            config.launchd_path.parent.mkdir(parents=True)
            config.launchd_path.write_bytes(b"operator-owned")
            with self.assertRaises(schedule.SchedulerError):
                schedule.install(config, apply=True)
            self.assertEqual(config.launchd_path.read_bytes(), b"operator-owned")

    def test_install_backs_up_owned_file_and_disable_is_reversible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            first = schedule.install(config, apply=True)
            self.assertTrue(first["applied"])
            original = config.launchd_path.read_bytes()
            config2 = schedule.SchedulerConfig(
                runtime_root=config.runtime_root,
                home=config.home,
                ace_bin=config.ace_bin,
                python_bin=config.python_bin,
                platform_name="macos",
                central_host=True,
            )
            schedule.install(config2, apply=True)
            self.assertTrue(list(config.launchd_path.parent.glob(f"{config.launchd_path.name}.bak*")))
            self.assertEqual(config.launchd_path.read_bytes(), original)
            preview = schedule.disable(config, apply=False)
            self.assertFalse(preview["applied"])
            applied = schedule.disable(config, apply=True)
            self.assertTrue(applied["disabled"])
            self.assertTrue(config.launchd_path.exists())

    def test_activate_uses_injected_subprocess_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            runner = Mock(return_value=type("Result", (), {"returncode": 0})())
            schedule.install(config, apply=True, activate=True, runner=runner)
            runner.assert_called()
            command = runner.call_args.args[0]
            self.assertEqual(command[0], str(schedule.LAUNCHCTL))


if __name__ == "__main__":
    unittest.main()
