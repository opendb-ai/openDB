"""Tests for the deterministic bi-temporal knowledge-graph logic.

Extraction uses an LLM, but parsing, invalidation, and rendering are pure and
deterministic — those are what we test here (no network, always run).
"""
from __future__ import annotations

from opendb_core.temporal_kg import Fact, parse_facts, bitemporalize, render_facts


def test_parse_pipe_facts():
    text = (
        "user | city | Boston | 2023-01-10\n"
        "garbage line without pipes\n"
        "user | 5k_pb | 27:12 | 2023-05-25\n"
        "- user | 5k_pb | 25:50 | 2023-05-27\n"
    )
    facts = parse_facts(text)
    assert len(facts) == 3
    assert facts[0].subject == "user" and facts[0].attribute == "city"
    assert facts[0].value == "Boston" and facts[0].valid_from == "2023-01-10"


def test_bitemporal_invalidation():
    facts = [
        Fact("user", "city", "Boston", "2023-01-10"),
        Fact("user", "city", "Seattle", "2023-06-01"),
        Fact("user", "5k_pb", "25:50", "2023-05-27"),
    ]
    out = bitemporalize(facts)
    by = {(f.subject, f.attribute, f.value): f for f in out}
    # Older value is closed off at the newer value's start; newest stays current.
    assert by[("user", "city", "Boston")].valid_to == "2023-06-01"
    assert by[("user", "city", "Seattle")].valid_to is None
    assert by[("user", "5k_pb", "25:50")].valid_to is None


def test_bitemporal_merges_consecutive_duplicates():
    facts = [
        Fact("user", "city", "Boston", "2023-01-10"),
        Fact("user", "city", "Boston", "2023-02-10"),  # restated, same value
        Fact("user", "city", "Seattle", "2023-06-01"),
    ]
    out = bitemporalize(facts)
    cities = [f for f in out if f.attribute == "city"]
    assert len(cities) == 2  # Boston (merged) + Seattle
    boston = next(f for f in cities if f.value == "Boston")
    assert boston.valid_from == "2023-01-10"  # earliest date kept
    assert boston.valid_to == "2023-06-01"


def test_render_shows_current_and_history():
    facts = bitemporalize([
        Fact("user", "city", "Boston", "2023-01-10"),
        Fact("user", "city", "Seattle", "2023-06-01"),
    ])
    rendered = render_facts(facts)
    assert "CURRENT = Seattle" in rendered
    assert "Boston" in rendered and "history" in rendered.lower()


def test_render_only_subjects_filter():
    facts = bitemporalize([
        Fact("user", "city", "Boston", "2023-01-10"),
        Fact("alice", "role", "manager", "2023-03-01"),
    ])
    rendered = render_facts(facts, only_subjects={"user"})
    assert "Boston" in rendered
    assert "alice" not in rendered
