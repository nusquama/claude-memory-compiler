"""Bounded, synthetic evaluation harness for ACE extraction signals.

The normal ACE runner intentionally has one fixed model contract.  This module
is a separate experiment harness: it never imports or changes that contract,
never points a child at the vault, and only permits the model/effort pairs
listed in :data:`MODEL_SPECS`.

By default the CLI prints a dry-run plan.  ``--run`` performs at most one
ephemeral, read-only Codex call per permitted pair and writes an aggregate
report outside the vault.  Reports contain synthetic case metadata and
aggregate metrics, never model transcripts or raw diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from utils import redact_sensitive_text, safe_codex_diagnostic_lines


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = SCRIPT_DIR.parent
VAULT_ROOT = CONFIG_ROOT.parent
DEFAULT_REPORT_DIR = Path.home() / ".agents" / "private" / "ace" / "evaluations"
DEFAULT_TIMEOUT_SECONDS = 120
MAX_CALLS = 3
CODEX_EXEC_PATH = os.environ.get("ACE_CODEX_EXEC_PATH", "codex")


@dataclass(frozen=True)
class ModelSpec:
    """An explicitly supported model/reasoning pair."""

    model: str
    effort: str

    @property
    def name(self) -> str:
        return f"{self.model}/{self.effort}"


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("gpt-5.6-luna", "low"),
    ModelSpec("gpt-5.6-luna", "medium"),
    ModelSpec("gpt-6-astra", "medium"),
)
ALLOWED_MODELS = tuple(dict.fromkeys(spec.model for spec in MODEL_SPECS))
ALLOWED_EFFORTS = tuple(dict.fromkeys(spec.effort for spec in MODEL_SPECS))


@dataclass(frozen=True)
class EvalCase:
    """A small synthetic labelled case; no user transcript is needed."""

    case_id: str
    finding_type: str
    expected_classification: str
    supported_finding: bool
    evidence_refs: tuple[str, ...]
    observation: str


@dataclass(frozen=True)
class InvocationResult:
    """Safe summary of a child call; the raw response is intentionally transient."""

    response: str | None
    latency_ms: float
    token_usage: dict[str, int] | None = None
    error_kind: str | None = None
    diagnostic_lines: tuple[str, ...] = ()
    error_fields: dict[str, str] | None = None


@dataclass(frozen=True)
class ParsedResponse:
    payload: dict[str, Any] | None
    structured_valid: bool
    error_kind: str | None = None


CLASSIFICATIONS: dict[str, frozenset[str]] = {
    "surengineering": frozenset({"justified", "unjustified", "none"}),
    "frustration": frozenset({"actual_mismatch", "quoted_language", "none"}),
    "tool_recovery": frozenset({"recovered", "not_recovered", "none"}),
    "verification": frozenset({"unverified", "verified", "none"}),
}


def build_cases() -> tuple[EvalCase, ...]:
    """Return the fixed, synthetic benchmark set.

    Evidence references are deliberately stable.  They let scoring reject a
    plausible-sounding label that is not grounded in the supplied observation.
    """

    return (
        EvalCase(
            "surengineering_justified",
            "surengineering",
            "justified",
            True,
            ("ev-sj-1", "ev-sj-2"),
            (
                "A maintenance task requires an idempotent migration over 50,000 rows. "
                "The agent adds bounded retries, a dry-run check, and a rollback note; "
                "each addition addresses an explicit failure mode in the task."
            ),
        ),
        EvalCase(
            "surengineering_unjustified",
            "surengineering",
            "unjustified",
            True,
            ("ev-su-1", "ev-su-2"),
            (
                "The request is to rename one local variable. The agent adds a plugin "
                "registry, a new service layer, three config files, and a deployment "
                "pipeline without any requirement or risk that calls for them."
            ),
        ),
        EvalCase(
            "frustration_actual_mismatch",
            "frustration",
            "actual_mismatch",
            True,
            ("ev-fm-1", "ev-fm-2"),
            (
                "The assistant says the export completed, but the requested output file "
                "is absent and the verification command returns a missing-file error. "
                "The user then says the result is not what was requested."
            ),
        ),
        EvalCase(
            "frustration_quoted_language",
            "frustration",
            "quoted_language",
            False,
            ("ev-fq-1",),
            (
                "A documentation example asks how to classify the quoted phrase "
                "'this is crap' when it appears in an old support ticket. It describes "
                "no complaint about the current run and no mismatch in the work."
            ),
        ),
        EvalCase(
            "failed_tool_recovered",
            "tool_recovery",
            "recovered",
            True,
            ("ev-tr-1", "ev-tr-2", "ev-tr-3"),
            (
                "The first file-read command fails because the path is wrong. The agent "
                "corrects the path, reruns the command successfully, and cites the "
                "returned contents in its final answer."
            ),
        ),
        EvalCase(
            "unverified_success",
            "verification",
            "unverified",
            True,
            ("ev-uv-1", "ev-uv-2"),
            (
                "The assistant reports that a configuration update is complete, but no "
                "post-change read, test, or destination query is shown. The observation "
                "does not establish that the update took effect."
            ),
        ),
        EvalCase(
            "clean_control",
            "none",
            "none",
            False,
            (),
            "A routine status note reports a completed local read and contains no failure, mismatch, or unsupported claim.",
        ),
    )


def build_output_schema() -> dict[str, Any]:
    """Return the strict JSON schema supplied to ``codex exec``."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["findings"],
        "properties": {
            "findings": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["case_id", "finding_type", "classification", "evidence_refs"],
                    "properties": {
                        "case_id": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                        "finding_type": {
                            "type": "string",
                            "enum": [
                                "surengineering",
                                "frustration",
                                "tool_recovery",
                                "verification",
                            ],
                        },
                        "classification": {"type": "string"},
                        "evidence_refs": {
                            "type": "array",
                            "items": {"type": "string", "pattern": "^ev-[a-z]+-[0-9]+$"},
                            "minItems": 1,
                        },
                    },
                },
            }
        },
    }


def build_prompt(cases: Sequence[EvalCase] | None = None) -> str:
    """Build one identical prompt for every model/effort pair."""

    selected_cases = tuple(cases or build_cases())
    observations = "\n\n".join(
        f"CASE {case.case_id}\nObservation: {case.observation}\n"
        f"Available evidence refs: {', '.join(case.evidence_refs) if case.evidence_refs else '(none)'}"
        for case in selected_cases
    )
    return (
        "Classify the synthetic ACE observations below. Return ONLY one JSON object, "
        "with a top-level `findings` array. Do not use Markdown fences, commentary, "
        "or fields outside the requested schema. Emit at most one finding per case "
        "and omit cases with no finding. Every finding must use one or more evidence "
        "refs listed for that same case; never invent a ref.\n\n"
        "Finding types and allowed classifications:\n"
        "- surengineering: justified or unjustified\n"
        "- frustration: actual_mismatch or quoted_language\n"
        "- tool_recovery: recovered or not_recovered\n"
        "- verification: unverified or verified\n\n"
        "A quoted phrase is not an actual frustration mismatch. A successful recovery "
        "is still a finding because the initial tool failure matters. A success claim "
        "without a post-change check is unverified. The evidence refs are synthetic "
        "and contain no private transcript data.\n\n"
        f"{observations}\n"
    )


def select_specs(model: str | None = None, effort: str | None = None) -> tuple[ModelSpec, ...]:
    """Select only explicitly permitted pairs; never infer a global winner."""

    selected = tuple(
        spec
        for spec in MODEL_SPECS
        if (model is None or spec.model == model) and (effort is None or spec.effort == effort)
    )
    if not selected:
        requested = f"model={model!r}, effort={effort!r}"
        raise ValueError(f"unsupported model/effort selection: {requested}")
    if len(selected) > MAX_CALLS:
        raise ValueError(f"bounded evaluation allows at most {MAX_CALLS} calls")
    return selected


def build_codex_command(
    spec: ModelSpec,
    *,
    cwd: Path,
    schema_path: Path,
    output_path: Path,
    codex_exec: str = CODEX_EXEC_PATH,
) -> list[str]:
    """Construct the read-only, non-recursive child invocation."""

    return [
        codex_exec,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--model",
        spec.model,
        "--config",
        f"model_reasoning_effort={spec.effort}",
        "--cd",
        str(cwd),
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "-",
    ]


def safe_child_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Pass the opaque CLI environment without logging values or secrets.

    Codex authentication may be carried by an environment variable unknown to
    this harness.  Copy the environment for the child, but never inspect,
    print, or persist its values.  The child still receives only the explicit
    recursion guards below.
    """

    source = dict(base if base is not None else os.environ)
    child_env = dict(source)
    child_env.update(
        {
            "CODEX_ACE_BACKFILL_ENABLED": "0",
            "CODEX_TURN_ENDED_FORWARD_SKY": "0",
            "CLAUDE_INVOKED_BY": "ace_evaluate",
            "ACE_FLUSH_ENGINE": "codex",
            "ACE_EVALUATION": "1",
        }
    )
    return child_env


def _extract_token_usage(stdout: str) -> dict[str, int] | None:
    """Extract usage only from explicit JSON counters, otherwise return None."""

    def walk(value: Any) -> Iterable[Mapping[str, Any]]:
        if isinstance(value, Mapping):
            yield value
            for nested in value.values():
                yield from walk(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from walk(nested)

    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        for mapping in walk(event):
            total = mapping.get("total_tokens")
            input_tokens = mapping.get("input_tokens", mapping.get("prompt_tokens"))
            output_tokens = mapping.get("output_tokens", mapping.get("completion_tokens"))
            if isinstance(total, int) and total >= 0:
                usage = {"total_tokens": total}
                if isinstance(input_tokens, int) and input_tokens >= 0:
                    usage["input_tokens"] = input_tokens
                if isinstance(output_tokens, int) and output_tokens >= 0:
                    usage["output_tokens"] = output_tokens
                return usage
            if (
                isinstance(input_tokens, int)
                and input_tokens >= 0
                and isinstance(output_tokens, int)
                and output_tokens >= 0
            ):
                return {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
    return None


def _safe_error_fields(stderr_text: str) -> dict[str, str] | None:
    """Extract only bounded, redacted ``type``, ``code``, and ``message`` fields."""

    # Redact before attempting to parse and cap the input so an unexpected CLI
    # trace cannot become an exported diagnostic.  The raw stderr never leaves
    # this function.
    bounded_parts = [stderr_text[:12000]]
    if len(stderr_text) > 12000:
        bounded_parts.append(stderr_text[-12000:])
    redacted = "\n".join(redact_sensitive_text(part) for part in bounded_parts)
    marker = re.search(r"(?im)^\s*error:\s*", redacted)
    if marker is None:
        return None
    candidate = redacted[marker.end() :].lstrip()
    parsed: Any = None
    if candidate.startswith("{"):
        try:
            parsed, _ = json.JSONDecoder().raw_decode(candidate)
        except (TypeError, ValueError):
            parsed = None

    fields: dict[str, str] = {}
    if isinstance(parsed, Mapping):
        for key in ("type", "code", "message"):
            value = parsed.get(key)
            if isinstance(value, (str, int, float, bool)):
                cleaned = " ".join(str(value).split())[:500]
                if cleaned:
                    fields[key] = cleaned
        nested_error = parsed.get("error")
        if isinstance(nested_error, Mapping):
            outer_type = fields.get("type")
            for key in ("code", "message"):
                value = nested_error.get(key)
                if key not in fields and isinstance(value, (str, int, float, bool)):
                    cleaned = " ".join(str(value).split())[:500]
                    if cleaned:
                        fields[key] = cleaned
            nested_type_value = nested_error.get("type")
            nested_type = None
            if isinstance(nested_type_value, (str, int, float, bool)):
                nested_type = " ".join(str(nested_type_value).split())[:500]
                if nested_type:
                    fields["type"] = nested_type
            if nested_type and outer_type and nested_type != outer_type:
                fields["outer_type"] = outer_type

    # If the provider's JSON is malformed, retain only individually bounded
    # scalar fields from the already-redacted text.
    for key in ("type", "code", "message"):
        if key in fields:
            continue
        match = re.search(rf'"{key}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', candidate)
        if match:
            cleaned = " ".join(match.group(1).replace('\\"', '"').split())[:500]
            if cleaned:
                fields[key] = cleaned
    return fields or None


def invoke_codex(
    spec: ModelSpec,
    prompt: str,
    *,
    cwd: Path,
    schema_path: Path,
    output_path: Path,
    timeout: int,
    codex_exec: str = CODEX_EXEC_PATH,
) -> InvocationResult:
    """Run one bounded child and discard all raw output after parsing it."""

    command = build_codex_command(
        spec,
        cwd=cwd,
        schema_path=schema_path,
        output_path=output_path,
        codex_exec=codex_exec,
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            cwd=str(cwd),
            env=safe_child_environment(),
            timeout=max(1, timeout),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return InvocationResult(None, (time.perf_counter() - started) * 1000, error_kind="timeout")
    except FileNotFoundError:
        return InvocationResult(None, (time.perf_counter() - started) * 1000, error_kind="cli_missing")
    except OSError:
        return InvocationResult(None, (time.perf_counter() - started) * 1000, error_kind="process_error")

    latency_ms = (time.perf_counter() - started) * 1000
    diagnostics = tuple(safe_codex_diagnostic_lines(completed.stderr or ""))
    error_fields = _safe_error_fields(completed.stderr or "")
    token_usage = _extract_token_usage(completed.stdout)
    if completed.returncode != 0:
        return InvocationResult(
            None,
            latency_ms,
            token_usage,
            error_kind="process_failed",
            diagnostic_lines=diagnostics,
            error_fields=error_fields,
        )

    response = ""
    if output_path.is_file():
        try:
            response = output_path.read_text(encoding="utf-8").strip()
        except OSError:
            response = ""
    if not response:
        response = completed.stdout.strip()
    if not response:
        return InvocationResult(
            None,
            latency_ms,
            token_usage,
            error_kind="empty_response",
            diagnostic_lines=diagnostics,
            error_fields=error_fields,
        )
    return InvocationResult(
        response,
        latency_ms,
        token_usage,
        diagnostic_lines=diagnostics,
        error_fields=error_fields,
    )


def parse_response(response: str | None, cases: Sequence[EvalCase]) -> ParsedResponse:
    """Parse strict JSON and validate semantic shape against the cases."""

    if not response:
        return ParsedResponse(None, False, "empty_response")
    try:
        payload = json.loads(response)
    except (TypeError, ValueError):
        return ParsedResponse(None, False, "invalid_json")
    if not isinstance(payload, dict):
        return ParsedResponse(None, False, "invalid_shape")
    if validate_payload(payload, cases):
        return ParsedResponse(payload, True)
    return ParsedResponse(payload, False, "schema_mismatch")


def validate_payload(payload: Mapping[str, Any], cases: Sequence[EvalCase]) -> bool:
    """Apply strict, case-aware validation after the CLI schema check."""

    if set(payload) != {"findings"} or not isinstance(payload.get("findings"), list):
        return False
    case_map = {case.case_id: case for case in cases}
    seen: set[str] = set()
    for finding in payload["findings"]:
        if not isinstance(finding, dict):
            return False
        if set(finding) != {"case_id", "finding_type", "classification", "evidence_refs"}:
            return False
        case_id = finding.get("case_id")
        finding_type = finding.get("finding_type")
        classification = finding.get("classification")
        evidence_refs = finding.get("evidence_refs")
        if not isinstance(case_id, str) or case_id not in case_map or case_id in seen:
            return False
        if not isinstance(finding_type, str) or finding_type not in CLASSIFICATIONS:
            return False
        if not isinstance(classification, str) or classification not in CLASSIFICATIONS[finding_type]:
            return False
        if not isinstance(evidence_refs, list) or not evidence_refs:
            return False
        if not all(isinstance(ref, str) for ref in evidence_refs) or len(set(evidence_refs)) != len(evidence_refs):
            return False
        if not set(evidence_refs).issubset(set(case_map[case_id].evidence_refs)):
            return False
        if finding_type != case_map[case_id].finding_type:
            return False
        seen.add(case_id)
    return True


def _finding_by_case(payload: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not payload or not isinstance(payload.get("findings"), list):
        return {}
    return {
        finding["case_id"]: finding
        for finding in payload["findings"]
        if isinstance(finding, dict) and isinstance(finding.get("case_id"), str)
    }


def score_response(
    parsed: ParsedResponse,
    cases: Sequence[EvalCase],
    *,
    latency_ms: float | None = None,
    token_usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compute grounded precision/recall plus validity and operational metrics."""

    payload = parsed.payload
    if payload is None:
        # A process failure, timeout, empty response, or malformed JSON is not
        # evidence that the model classified any case. Keep the operational
        # failure visible while leaving prediction metrics explicitly N/A.
        return {
            "structured_validity": 0.0,
            "structured_valid": False,
            "classification_accuracy": None,
            "evidence_grounding_rate": None,
            "supported_findings": {
                "gold": sum(case.supported_finding for case in cases),
                "predicted": None,
                "true_positive": None,
                "false_positive": None,
                "false_negative": None,
                "precision": None,
                "recall": None,
                "f1": None,
            },
            "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
            "token_usage": token_usage,
            "token_usage_measurable": token_usage is not None,
            "parse_error": parsed.error_kind,
        }
    predictions = _finding_by_case(payload)
    case_map = {case.case_id: case for case in cases}
    true_positive = 0
    false_positive = 0
    false_negative = 0
    label_correct = 0
    evidence_grounded = 0
    gold_supported = sum(case.supported_finding for case in cases)

    for case in cases:
        prediction = predictions.get(case.case_id)
        if prediction is None:
            if case.expected_classification == "none":
                label_correct += 1
            elif case.supported_finding:
                false_negative += 1
            continue

        same_label = (
            prediction.get("finding_type") == case.finding_type
            and prediction.get("classification") == case.expected_classification
        )
        refs = prediction.get("evidence_refs")
        grounded = (
            isinstance(refs, list)
            and bool(refs)
            and set(refs).issubset(set(case.evidence_refs))
            and bool(set(refs).intersection(case.evidence_refs))
        )
        if same_label:
            label_correct += 1
        if grounded:
            evidence_grounded += 1

        is_positive_prediction = prediction.get("classification") not in {"none", "quoted_language"}
        exact_supported = case.supported_finding and same_label and grounded
        if exact_supported:
            true_positive += 1
        elif case.supported_finding:
            false_negative += 1
            if is_positive_prediction:
                false_positive += 1
        elif is_positive_prediction:
            false_positive += 1

    # Unknown case IDs are structurally invalid, but still count as false
    # positives if a caller scores a partially parsed response.
    for case_id, prediction in predictions.items():
        if case_id not in case_map and prediction.get("classification") not in {"none", "quoted_language"}:
            false_positive += 1

    predicted_supported = true_positive + false_positive
    precision = true_positive / predicted_supported if predicted_supported else 0.0
    recall = true_positive / gold_supported if gold_supported else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "structured_validity": 1.0 if parsed.structured_valid else 0.0,
        "structured_valid": parsed.structured_valid,
        "classification_accuracy": label_correct / len(cases) if cases else 0.0,
        "evidence_grounding_rate": evidence_grounded / len(predictions) if predictions else 0.0,
        "supported_findings": {
            "gold": gold_supported,
            "predicted": predicted_supported,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        # None is intentional: the normal invocation does not request usage
        # events, and the harness must never invent token counts.
        "token_usage": token_usage,
        "token_usage_measurable": token_usage is not None,
        "parse_error": parsed.error_kind,
    }


def _assert_report_outside_vault(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    vault = VAULT_ROOT.resolve()
    if resolved == vault or resolved.is_relative_to(vault):
        raise ValueError("evaluation reports must be outside the ACE vault")
    return resolved


def write_report(report: Mapping[str, Any], output_dir: Path) -> Path:
    """Write a private aggregate report and no raw model trace."""

    destination = _assert_report_outside_vault(output_dir)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        destination.chmod(0o700)
    except OSError:
        pass
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = destination / f"ace-evaluation-{stamp}.json"
    # Avoid a same-second overwrite if two explicitly requested runs finish
    # together; this remains bounded and does not scan or touch the vault.
    suffix = 1
    while report_path.exists():
        report_path = destination / f"ace-evaluation-{stamp}-{suffix}.json"
        suffix += 1
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        report_path.chmod(0o600)
    except OSError:
        pass
    return report_path


Invoker = Callable[..., InvocationResult]


def run_evaluation(
    specs: Sequence[ModelSpec],
    *,
    cases: Sequence[EvalCase] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    output_dir: Path = DEFAULT_REPORT_DIR,
    codex_exec: str = CODEX_EXEC_PATH,
    invoker: Invoker | None = None,
) -> Path:
    """Run the bounded experiment and write a private, transcript-free report."""

    selected_cases = tuple(cases or build_cases())
    selected_specs = tuple(specs)
    if not selected_specs or len(selected_specs) > MAX_CALLS:
        raise ValueError(f"run requires 1 to {MAX_CALLS} model/effort pairs")
    if len(set(selected_specs)) != len(selected_specs):
        raise ValueError("run cannot repeat a model/effort pair")
    if any(spec not in MODEL_SPECS for spec in selected_specs):
        raise ValueError("run contains an unsupported model/effort pair")
    prompt = build_prompt(selected_cases)
    schema = build_output_schema()
    runs: list[dict[str, Any]] = []
    call = invoker or invoke_codex

    with tempfile.TemporaryDirectory(prefix="ace-evaluation-") as temp:
        work_dir = Path(temp)
        schema_path = work_dir / "schema.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        for index, spec in enumerate(selected_specs, start=1):
            output_path = work_dir / f"last-message-{index}.json"
            try:
                invocation = call(
                    spec,
                    prompt,
                    cwd=work_dir,
                    schema_path=schema_path,
                    output_path=output_path,
                    timeout=timeout,
                    codex_exec=codex_exec,
                )
            except Exception:
                # Keep the report useful without exporting exception text that
                # could contain command lines, paths, or echoed input.
                invocation = InvocationResult(None, 0.0, error_kind="invoker_error")
            parsed = parse_response(invocation.response, selected_cases)
            metrics = score_response(
                parsed,
                selected_cases,
                latency_ms=invocation.latency_ms,
                token_usage=invocation.token_usage,
            )
            status = "ok" if parsed.structured_valid else (
                "error" if invocation.error_kind else "invalid_output"
            )
            runs.append(
                {
                    "model": spec.model,
                    "effort": spec.effort,
                    "status": status,
                    "error_kind": invocation.error_kind or parsed.error_kind,
                    "safe_diagnostics": list(invocation.diagnostic_lines),
                    "safe_error_fields": invocation.error_fields,
                    "metrics": metrics,
                }
            )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": {
            "purpose": "synthetic ACE extraction comparison",
            "dry_run": False,
            "max_calls": MAX_CALLS,
            "calls": len(runs),
            "timeout_seconds": timeout,
            "codex_exec": codex_exec,
            "vault_write": False,
            "child_ephemeral": True,
            "child_read_only": True,
            "child_ignores_user_config": True,
            "child_ignores_rules": True,
            "recursion_disabled": True,
        },
        "cases": [
            {
                "case_id": case.case_id,
                "finding_type": case.finding_type,
                "expected_classification": case.expected_classification,
                "supported_finding": case.supported_finding,
                "evidence_refs": list(case.evidence_refs),
            }
            for case in selected_cases
        ],
        "runs": runs,
        "interpretation": (
            "This is a bounded synthetic sample. Metrics compare the listed pairs only; "
            "they do not select or change ACE's global model contract. Token usage is "
            "reported only when explicit CLI counters are measurable."
        ),
    }
    return write_report(report, output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="perform the bounded real experiment")
    parser.add_argument("--model", choices=ALLOWED_MODELS, help="optional model filter")
    parser.add_argument("--effort", choices=ALLOWED_EFFORTS, help="optional reasoning-effort filter")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"per-call timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="private report directory (default: ~/.agents/private/ace/evaluations)",
    )
    parser.add_argument(
        "--codex-exec",
        default=CODEX_EXEC_PATH,
        help=(
            "Codex executable override; prefer ACE_CODEX_EXEC_PATH for an explicit path "
            f"(default: {CODEX_EXEC_PATH})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        specs = select_specs(args.model, args.effort)
    except ValueError as exc:
        parser.error(str(exc))
    if args.timeout < 1:
        parser.error("--timeout must be at least 1 second")

    if not args.run:
        print("ACE evaluation dry-run plan (no model call, no report, no vault write)")
        print(f"Synthetic cases: {len(build_cases())}")
        print(f"Calls: {len(specs)} (maximum {MAX_CALLS})")
        for spec in specs:
            print(f"- {spec.name}")
        print(f"Report target on --run: {_assert_report_outside_vault(args.output_dir)}")
        return 0

    try:
        report_path = run_evaluation(
            specs,
            timeout=args.timeout,
            output_dir=args.output_dir,
            codex_exec=args.codex_exec,
        )
    except (OSError, ValueError) as exc:
        print(f"ACE evaluation could not complete: {type(exc).__name__}", file=__import__("sys").stderr)
        return 1
    print(f"ACE evaluation report: {report_path}")
    print("Limited synthetic benchmark complete; no global model selection was applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
