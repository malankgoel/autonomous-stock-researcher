"""Persistent, single-use locked-holdout enforcement."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from typing import Any


class HoldoutExhaustedError(RuntimeError):
    """Raised when code attempts to touch a spent holdout."""


class HoldoutManager:
    def __init__(self, config: dict, state_path: str) -> None:
        holdout = config.get("holdout", config)
        self.start_date = date.fromisoformat(str(holdout["start_date"]))
        self.end_date = date.fromisoformat(str(holdout["end_date"]))
        self.initial_uses = int(holdout.get("uses_remaining", 1))
        if self.initial_uses != 1:
            raise ValueError("the registered holdout must permit exactly one use")
        self.state_path = Path(state_path)

    def _read_state(self, handle) -> dict:
        handle.seek(0)
        text = handle.read()
        if not text:
            return {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "uses_remaining": self.initial_uses,
            }
        state = json.loads(text)
        if (state.get("start_date"), state.get("end_date")) != (
            self.start_date.isoformat(),
            self.end_date.isoformat(),
        ):
            raise ValueError("persistent state belongs to a different holdout interval")
        return state

    @property
    def uses_remaining(self) -> int:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            return int(self._read_state(handle)["uses_remaining"])

    def evaluate_once(self, evaluator: Callable[..., Any], *args, **kwargs):
        """Consume the holdout durably, then invoke ``evaluator`` exactly once.

        Dates are available as manager attributes; arguments are forwarded
        unchanged so callers must explicitly wire the registered range into the
        harness.  Consumption happens before evaluation, including failed trials.
        """
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state = self._read_state(handle)
            if int(state["uses_remaining"]) <= 0:
                raise HoldoutExhaustedError("locked holdout has already been used")
            state["uses_remaining"] = int(state["uses_remaining"]) - 1
            state["consumed_at_utc"] = datetime.now(timezone.utc).isoformat()
            handle.seek(0)
            json.dump(state, handle, sort_keys=True)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
        return evaluator(*args, **kwargs)
