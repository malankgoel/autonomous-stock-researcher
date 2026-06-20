from __future__ import annotations

import json
from datetime import date

import pytest

import library.edge_library as edge_module
from backtest.labels import BacktestResult, SignalResult
from hypothesis.spec import Direction, HypothesisSpec
from library.edge_library import DuplicateEdgeError, EdgeLibrary, MalformedLibraryError


def _spec(edge_id: str = "edge_b") -> HypothesisSpec:
    return HypothesisSpec(
        id=edge_id,
        description="A validated test edge",
        source="human",
        tier=1,
        generation_batch="batch-7",
        universe_filter={"min_dollar_volume": 1_000_000, "cap": "any"},
        entry_condition={"momentum": {">": 0.1}},
        direction=Direction.LONG,
        horizon_days=20,
        entry_timing="next_open",
        exit_rule={"horizon": 20, "stop": -0.1},
        features=["momentum", "open", "close"],
    )


def _result(spec: HypothesisSpec) -> BacktestResult:
    signal = SignalResult(
        spec_id=spec.id,
        ticker="TEST",
        signal_date=date(2018, 1, 2),
        entry_date=date(2018, 1, 3),
        entry_price=20.0,
        direction="long",
        forward_returns={20: 0.08},
        sector_relative_returns={20: 0.06},
        cost_return=0.002,
        exit_date=date(2018, 1, 31),
        exit_reason="horizon",
    )
    return BacktestResult(
        spec_id=spec.id,
        generation_batch=spec.generation_batch,
        signals=[signal],
        universe_size=800,
        start_date=date(2014, 1, 1),
        end_date=date(2019, 12, 31),
    )


def _add(library: EdgeLibrary, spec: HypothesisSpec, **kwargs) -> None:
    library.add(
        spec,
        _result(spec),
        {"deflated_sharpe": 0.97, "n_trials": 12},
        modeled_costs={"round_trip_bps": 20.0, "model": "square_root"},
        capacity_estimate={"daily_notional_usd": 250_000, "max_adv_fraction": 0.05},
        tested_regimes=["bull", "bear"],
        holdout_status={"evaluated": True, "passed": True},
        provenance={"code_revision": "test-revision", "data_snapshot": "synthetic-v1"},
        **kwargs,
    )


def test_round_trip_persists_complete_audit_evidence(tmp_path):
    path = tmp_path / "edges.json"
    spec = _spec()
    _add(EdgeLibrary(path), spec)

    record = EdgeLibrary(path).all()[0]
    assert record["spec"] == spec.to_dict()
    assert record["result"]["signals"][0]["signal_date"] == "2018-01-02"
    assert record["statistics"]["deflated_sharpe"] == 0.97
    assert record["modeled_costs"]["round_trip_bps"] == 20.0
    assert record["capacity_estimate"]["daily_notional_usd"] == 250_000
    assert record["tested_regimes"] == ["bull", "bear"]
    assert record["generation_batch"] == "batch-7"
    assert record["holdout_status"] == {"evaluated": True, "passed": True}
    assert len(record["provenance"]["evidence_sha256"]) == 64
    assert json.loads(path.read_text())["schema_version"] == 1


def test_duplicate_rejected_and_explicit_replacement_supported(tmp_path):
    library = EdgeLibrary(tmp_path / "edges.json")
    spec = _spec()
    _add(library, spec)
    with pytest.raises(DuplicateEdgeError, match=spec.id):
        _add(library, spec)

    library.add(spec, _result(spec), {"deflated_sharpe": 0.99}, replace=True)
    assert len(library.all()) == 1
    assert library.all()[0]["statistics"]["deflated_sharpe"] == 0.99


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        '{"schema_version": 99, "edges": []}',
        '{"schema_version": 1, "edges": [{"id": "incomplete"}]}',
    ],
)
def test_malformed_storage_is_rejected(tmp_path, payload):
    path = tmp_path / "edges.json"
    path.write_text(payload)
    with pytest.raises(MalformedLibraryError):
        EdgeLibrary(path).all()


def test_tampered_nested_evidence_is_rejected(tmp_path):
    path = tmp_path / "edges.json"
    _add(EdgeLibrary(path), _spec())
    document = json.loads(path.read_text())
    document["edges"][0]["result"]["signals"][0]["forward_returns"]["20"] = 99.0
    path.write_text(json.dumps(document))
    with pytest.raises(MalformedLibraryError, match="digest mismatch"):
        EdgeLibrary(path).all()


def test_failed_atomic_replace_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "edges.json"
    library = EdgeLibrary(path)
    _add(library, _spec("edge_a"))
    before = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(edge_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        _add(library, _spec("edge_b"))
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_all_is_deterministically_ordered_by_edge_id(tmp_path):
    library = EdgeLibrary(tmp_path / "edges.json")
    for edge_id in ("edge_c", "edge_a", "edge_b"):
        _add(library, _spec(edge_id))
    assert [record["id"] for record in library.all()] == ["edge_a", "edge_b", "edge_c"]
