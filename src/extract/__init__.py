"""Tier-2 text-feature extraction layer.

The extraction LLM turns a piece of historical text (an EDGAR filing, a
transcript, a news item) into *objective, verifiable* features — never a
forward-looking judgement. See ``docs/tier2_scoping_brief.md`` for the contract
and the two leakage failure modes this layer is designed against.
"""
