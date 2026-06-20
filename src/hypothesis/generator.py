"""LLM hypothesis generator.

OWNER: Phase 1, Agent C (Universe + Compiler).
STATUS: DEFERRED. Do not build until the harness reproduces known anomalies
(README "When data arrives", step 3). Stub only.

Contract when built: given the spec schema and the available feature catalog, the
LLM proposes a list of valid, falsifiable ``HypothesisSpec`` objects (structural
and, later, qualitative). The generator NEVER sees outcomes. It proposes; the
deterministic harness and survival filter evaluate. Every batch is tagged with a
``generation_batch`` so the survival filter can count every test honestly.
"""

from __future__ import annotations

from hypothesis.spec import HypothesisSpec


def generate(prompt_context: dict, n: int, generation_batch: str) -> list[HypothesisSpec]:
    raise NotImplementedError("Agent C: deferred until anomaly reproduction passes.")
