from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import ace_migrate  # noqa: E402


class AceMigrationTests(unittest.TestCase):
    def test_default_state_consumer_uses_active_private_ace_root(self) -> None:
        self.assertEqual(
            ace_migrate.DEFAULT_STATE_DESTINATION,
            Path.home() / ".agents" / "private" / "ace",
        )
        parsed = ace_migrate._parser().parse_args([])
        self.assertEqual(
            Path(parsed.state_destination),
            Path.home() / ".agents" / "private" / "ace",
        )

    def _roots(self, root: Path) -> dict[str, Path]:
        return {
            "state_source": root / "old-state",
            "state_destination": root / "new-state",
            "private_report_source": root / "old-private",
            "private_report_destination": root / "new-private",
            "daily_report_source": root / "old-daily",
            "daily_report_destination": root / "new-daily",
            "evaluation_source": root / "old-evaluations",
            "evaluation_destination": root / "new-evaluations",
            "overengineering_source": root / "old-overengineering",
            "overengineering_destination": root / "new-overengineering",
        }

    def test_dry_run_is_default_and_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._roots(root)
            source = paths["state_source"] / "incident-tracking.json"
            source.parent.mkdir()
            source.write_bytes(b'{"decision":"keep"}\n')

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = ace_migrate.main(
                    [str_flag for pair in paths.items() for str_flag in (f"--{pair[0].replace('_', '-')}", str(pair[1]))]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(source.exists())
            self.assertFalse(paths["state_destination"].exists())
            self.assertIn("ACE migration dry-run", output.getvalue())

    def test_apply_copies_opaque_state_and_reports_without_deleting_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._roots(root)
            state = paths["state_source"] / "checkpoint-state.json"
            state.parent.mkdir()
            state.write_bytes(b'{"checkpoint": [1, 2]}\n')
            report = paths["private_report_source"] / "daily" / "2026-09-07.md"
            report.parent.mkdir(parents=True)
            report.write_bytes(b"# ACE report\n")
            lock = paths["state_source"] / "collection-state.lock"
            lock.write_bytes(b"active lock")

            args = ["--apply"]
            for key, value in paths.items():
                args.extend((f"--{key.replace('_', '-')}", str(value)))
            self.assertEqual(ace_migrate.main(args), 0)

            self.assertEqual(
                state.read_bytes(),
                (paths["state_destination"] / state.name).read_bytes(),
            )
            self.assertEqual(
                report.read_bytes(),
                (paths["private_report_destination"] / "daily" / report.name).read_bytes(),
            )
            self.assertTrue(state.exists())
            self.assertTrue(report.exists())
            self.assertFalse((paths["state_destination"] / lock.name).exists())

    def test_apply_is_idempotent_and_never_overwrites_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._roots(root)
            source = paths["state_source"] / "collection-state.json"
            source.parent.mkdir()
            source.write_bytes(b"source-state")

            args = ["--apply"]
            for key, value in paths.items():
                args.extend((f"--{key.replace('_', '-')}", str(value)))
            self.assertEqual(ace_migrate.main(args), 0)
            destination = paths["state_destination"] / source.name
            self.assertEqual(destination.read_bytes(), b"source-state")

            # An identical destination is safe and remains untouched.
            plan = ace_migrate.build_plan(**paths)
            self.assertEqual([item.status for item in plan], ["identical"])
            self.assertEqual(ace_migrate.apply_plan(plan)["copied"], 0)

            # A divergent destination is a collision, never an overwrite.
            destination.write_bytes(b"newer-destination")
            plan = ace_migrate.build_plan(**paths)
            self.assertEqual([item.status for item in plan], ["collision"])
            result = ace_migrate.apply_plan(plan)
            self.assertEqual(result["collisions"], 1)
            self.assertEqual(destination.read_bytes(), b"newer-destination")
            self.assertEqual(source.read_bytes(), b"source-state")

    def test_report_collision_uses_recoverable_archive_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._roots(root)
            source = paths["private_report_source"] / "daily" / "2026-09-07.md"
            destination = paths["private_report_destination"] / "daily" / source.name
            source.parent.mkdir(parents=True)
            destination.parent.mkdir(parents=True)
            source.write_bytes(b"legacy report")
            destination.write_bytes(b"newer report")

            plan = ace_migrate.build_plan(**paths)
            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0].status, "archive")
            self.assertEqual(
                plan[0].destination,
                paths["private_report_destination"] / "archive" / "cmc" / "daily" / source.name,
            )

            result = ace_migrate.apply_plan(plan)
            self.assertEqual(result["copied"], 1)
            self.assertEqual(destination.read_bytes(), b"newer report")
            self.assertEqual(plan[0].destination.read_bytes(), b"legacy report")


if __name__ == "__main__":
    unittest.main()
