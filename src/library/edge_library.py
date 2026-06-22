"""Persistent storage for validated, surviving edges.

The local format is one UTF-8 JSON file::

    {"schema_version": 1, "edges": [EDGE_RECORD, ...]}

Each record contains the validated ``HypothesisSpec``, complete serialized
``BacktestResult`` evidence, validation statistics, modeled costs, capacity,
tested regimes, holdout status, and audit provenance. Records are ordered by
spec ID and writes use a temporary file in the store directory followed by an
atomic replacement. The format is deliberately plain JSON so an audit does not
depend on this package to inspect the evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from backtest.labels import BacktestResult
from backtest.labels import SignalResult
from hypothesis.spec import HypothesisSpec, validate


SCHEMA_VERSION = 1


class EdgeLibraryError(ValueError):
    """Base error for invalid edge-library operations."""


class DuplicateEdgeError(EdgeLibraryError):
    """Raised when an edge ID already exists and replacement was not requested."""


class MalformedLibraryError(EdgeLibraryError):
    """Raised when persisted storage does not conform to the library schema."""


def _json_value(value: Any) -> Any:
    """Convert frozen-contract values to strict, deterministic JSON values."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MalformedLibraryError(message)


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _validate_result(result: BacktestResult, spec: HypothesisSpec) -> None:
    _require(isinstance(result, BacktestResult), "result must be a BacktestResult")
    _require(result.spec_id == spec.id, "result.spec_id must equal spec.id")
    _require(
        result.generation_batch == spec.generation_batch,
        "result.generation_batch must equal spec.generation_batch",
    )
    _require(result.n_signals == len(result.signals), "result.n_signals does not match signals")
    for index, signal in enumerate(result.signals):
        _require(signal.spec_id == spec.id, f"signal {index} spec_id does not match spec.id")
        _require(signal.direction in {"long", "short", "neutral"}, f"signal {index} direction")
        _require(signal.entry_date >= signal.signal_date, f"signal {index} entry precedes signal")
        if signal.exit_date is not None:
            _require(signal.exit_date >= signal.entry_date, f"signal {index} exit precedes entry")


def _validate_record(record: Any) -> None:
    _require(isinstance(record, dict), "edge record must be an object")
    expected = {
        "id",
        "spec",
        "result",
        "statistics",
        "modeled_costs",
        "capacity_estimate",
        "tested_regimes",
        "generation_batch",
        "holdout_status",
        "provenance",
    }
    _require(set(record) == expected, "edge record has missing or unknown fields")
    _require(isinstance(record["id"], str) and record["id"], "edge id must be non-empty")
    _require(isinstance(record["spec"], dict), "spec must be an object")
    try:
        spec = HypothesisSpec.from_dict(record["spec"])
        validate(spec)
    except (TypeError, ValueError) as exc:
        raise MalformedLibraryError(f"invalid persisted spec: {exc}") from exc
    _require(spec.id == record["id"], "record id does not match spec id")
    _require(record["generation_batch"] == spec.generation_batch, "generation batch mismatch")

    result = record["result"]
    _require(isinstance(result, dict), "result must be an object")
    result_fields = set(BacktestResult.__dataclass_fields__)  # type: ignore[attr-defined]
    _require(set(result) == result_fields, "result has missing or unknown fields")
    _require(result["spec_id"] == spec.id, "persisted result spec_id mismatch")
    _require(result["generation_batch"] == spec.generation_batch, "result batch mismatch")
    _require(isinstance(result["signals"], list), "result signals must be a list")
    _require(result["n_signals"] == len(result["signals"]), "result signal count mismatch")
    _require(isinstance(result["is_holdout"], bool), "result is_holdout must be a bool")
    signal_fields = set(SignalResult.__dataclass_fields__)  # type: ignore[attr-defined]
    for index, signal in enumerate(result["signals"]):
        _require(isinstance(signal, dict), f"signal {index} must be an object")
        _require(set(signal) == signal_fields, f"signal {index} has missing or unknown fields")
        _require(signal["spec_id"] == spec.id, f"signal {index} spec_id mismatch")
        _require(
            signal["direction"] in {"long", "short", "neutral"},
            f"signal {index} has an invalid direction",
        )
        for field in (
            "forward_returns",
            "sector_relative_returns",
            "smallcap_relative_returns",
            "market_relative_returns",
        ):
            _require(isinstance(signal[field], dict), f"signal {index} {field} must be an object")

    for field in (
        "statistics",
        "modeled_costs",
        "capacity_estimate",
        "holdout_status",
        "provenance",
    ):
        _require(isinstance(record[field], dict), f"{field} must be an object")
    _require(
        isinstance(record["tested_regimes"], list)
        and all(isinstance(item, str) and item for item in record["tested_regimes"]),
        "tested_regimes must contain non-empty strings",
    )
    _require(
        isinstance(record["holdout_status"].get("evaluated"), bool),
        "holdout_status.evaluated must be a bool",
    )
    passed = record["holdout_status"].get("passed")
    _require(
        passed is None or isinstance(passed, bool), "holdout_status.passed must be bool or null"
    )
    _require(_is_json_value(record), "record contains a non-JSON or non-finite value")
    digest = record["provenance"].get("evidence_sha256")
    _require(
        isinstance(digest, str) and len(digest) == 64,
        "provenance.evidence_sha256 must be a SHA-256 hex digest",
    )
    expected_digest = hashlib.sha256(
        json.dumps(
            {"spec": record["spec"], "result": result},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _require(digest == expected_digest, "persisted evidence digest mismatch")


def _validate_document(document: Any) -> None:
    _require(isinstance(document, dict), "library document must be an object")
    _require(set(document) == {"schema_version", "edges"}, "invalid library document fields")
    _require(document["schema_version"] == SCHEMA_VERSION, "unsupported schema_version")
    _require(isinstance(document["edges"], list), "edges must be a list")
    for record in document["edges"]:
        _validate_record(record)
    ids = [record["id"] for record in document["edges"]]
    _require(len(ids) == len(set(ids)), "duplicate edge IDs in storage")


class EdgeLibrary:
    """A versioned, atomic local repository of validated edge evidence."""

    def __init__(self, store_path: str | os.PathLike[str]) -> None:
        self.store_path = Path(store_path)

    def _read(self) -> dict[str, Any]:
        if not self.store_path.exists():
            return {"schema_version": SCHEMA_VERSION, "edges": []}
        try:
            document = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MalformedLibraryError(f"cannot read edge library: {exc}") from exc
        _validate_document(document)
        return document

    def _write(self, document: dict[str, Any]) -> None:
        _validate_document(document)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                document,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.store_path.parent,
                prefix=f".{self.store_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.store_path)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def add(
        self,
        spec: HypothesisSpec,
        result: BacktestResult,
        stats: dict[str, Any],
        *,
        modeled_costs: dict[str, Any] | None = None,
        capacity_estimate: dict[str, Any] | None = None,
        tested_regimes: list[str] | None = None,
        holdout_status: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        replace: bool = False,
    ) -> None:
        """Add an edge, rejecting duplicate IDs unless ``replace=True``.

        Optional evidence defaults are explicit ``not_estimated``/empty values;
        missing analysis is therefore distinguishable from an estimate of zero.
        """
        validate(spec)
        _validate_result(result, spec)
        _require(isinstance(stats, dict), "stats must be an object")

        costs = (
            modeled_costs
            if modeled_costs is not None
            else {
                "status": "summarized_from_signals",
                "mean_cost_return": (
                    sum(signal.cost_return for signal in result.signals) / len(result.signals)
                    if result.signals
                    else 0.0
                ),
            }
        )
        capacity = (
            capacity_estimate if capacity_estimate is not None else {"status": "not_estimated"}
        )
        regimes = tested_regimes if tested_regimes is not None else []
        holdout = (
            holdout_status
            if holdout_status is not None
            else {
                "evaluated": result.is_holdout,
                "passed": None,
            }
        )
        audit = {
            **(provenance or {}),
            "stored_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_start_date": _json_value(result.start_date),
            "result_end_date": _json_value(result.end_date),
            "universe_size": result.universe_size,
        }
        result_data = _json_value(asdict(result))
        digest_payload = {"spec": spec.to_dict(), "result": result_data}
        audit["evidence_sha256"] = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        record = {
            "id": spec.id,
            "spec": spec.to_dict(),
            "result": result_data,
            "statistics": _json_value(stats),
            "modeled_costs": _json_value(costs),
            "capacity_estimate": _json_value(capacity),
            "tested_regimes": _json_value(regimes),
            "generation_batch": spec.generation_batch,
            "holdout_status": _json_value(holdout),
            "provenance": _json_value(audit),
        }
        _validate_record(record)

        document = self._read()
        existing = next((item for item in document["edges"] if item["id"] == spec.id), None)
        if existing is not None and not replace:
            raise DuplicateEdgeError(f"edge ID already exists: {spec.id}")
        document["edges"] = [item for item in document["edges"] if item["id"] != spec.id]
        document["edges"].append(record)
        document["edges"].sort(key=lambda item: item["id"])
        self._write(document)

    def all(self) -> list[dict[str, Any]]:
        """Return schema-validated edge records in deterministic ID order."""
        records = self._read()["edges"]
        return sorted(records, key=lambda item: item["id"])
