"""CodeMemEval-hard: the methodology harness has to be trustworthy itself.

These tests exist because the module's whole claim is that it replaces
assertions with numbers. A paraphrase set that silently covers half the
questions, or a judge validation that reports 0% false-accept because every
call errored, would produce numbers that are worse than no numbers -- they
carry the authority of a measurement without being one.

The judge tests stub the model call. Running them must not need a paid
credential, for the same reason `_load_group` parses `gen_codemem` instead of
importing it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmark"
sys.path.insert(0, str(BENCH))

codemem_hard = pytest.importorskip("codemem_hard")


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

class TestWilson:
    def test_interval_brackets_the_point_estimate(self) -> None:
        lo, hi = codemem_hard.wilson(26, 27)
        assert lo < 26 / 27 < hi

    def test_stays_inside_zero_one_at_the_boundaries(self) -> None:
        """The reason for Wilson over the normal approximation: 27/27 must not
        produce an upper bound above 1.0, and 0/27 not a lower bound below 0."""
        lo, hi = codemem_hard.wilson(27, 27)
        assert 0.0 <= lo < hi <= 1.0
        lo, hi = codemem_hard.wilson(0, 27)
        assert 0.0 <= lo < hi <= 1.0

    def test_interval_narrows_as_n_grows(self) -> None:
        wide = codemem_hard.wilson(26, 27)
        narrow = codemem_hard.wilson(2600, 2700)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_zero_n_does_not_divide_by_zero(self) -> None:
        assert codemem_hard.wilson(0, 0) == (0.0, 0.0)

    def test_required_n_grows_as_the_interval_tightens(self) -> None:
        assert (codemem_hard.n_for_precision(0.95, 0.10)
                < codemem_hard.n_for_precision(0.95, 0.05)
                < codemem_hard.n_for_precision(0.95, 0.02))


# ----------------------------------------------------------------------
# Paraphrase coverage
# ----------------------------------------------------------------------

class TestParaphraseCoverage:
    def test_every_question_has_a_restatement(self) -> None:
        """A partial split scored and quoted as 'the paraphrase result' is the
        small-n error this module exists to name."""
        index = codemem_hard._evidence_index()
        missing = sorted(set(index) - set(codemem_hard.PARAPHRASE_QUESTIONS))
        assert not missing, f"no paraphrase for: {missing}"

    def test_no_paraphrase_targets_an_unknown_question(self) -> None:
        index = codemem_hard._evidence_index()
        unknown = sorted(set(codemem_hard.PARAPHRASE_QUESTIONS) - set(index))
        assert not unknown, f"paraphrase for nonexistent question: {unknown}"

    def test_restatements_share_no_content_words_with_the_evidence(self) -> None:
        """The point of the split is that lexical overlap is gone. If a
        paraphrase leaks evidence tokens it measures the same thing as the
        original question, and the comparison stops meaning anything."""
        offenders = []
        for qid, item in codemem_hard._evidence_index().items():
            if not item["evidence"]:
                continue  # abstention: no evidence to overlap with
            shared = (codemem_hard._tokens(codemem_hard.PARAPHRASE_QUESTIONS[qid])
                      & codemem_hard._tokens(item["evidence"]))
            if shared:
                offenders.append((qid, sorted(shared)))
        assert not offenders, f"paraphrases leaking evidence tokens: {offenders}"

    def test_original_questions_do_overlap(self) -> None:
        """The control. If the originals had no overlap either, the split would
        be measuring noise rather than the effect being claimed."""
        report = codemem_hard.report_overlap()
        mean_original = sum(r["coverage"] for r in report["original"]) / len(report["original"])
        assert mean_original > 0.3


# ----------------------------------------------------------------------
# Dataset emission
# ----------------------------------------------------------------------

class TestEmitParaphrase:
    def _dataset(self, tmp_path: Path, ids: list[str]) -> Path:
        src = tmp_path / "src.json"
        src.write_text(json.dumps([
            {"question_id": qid, "question": f"original text for {qid}",
             "question_type": "code-architecture",
             "haystack_session_ids": [f"{qid}_s0"], "haystack_sessions": [[]],
             "answer_session_ids": [f"{qid}_s0"], "answer": "gold"}
            for qid in ids
        ]))
        return src

    def test_only_the_question_text_changes(self, tmp_path: Path) -> None:
        """Same haystack, same gold, same dates -- so a score difference is
        attributable to the wording and nothing else."""
        src = self._dataset(tmp_path, ["arch_auth", "conv_error"])
        dest = tmp_path / "out.json"
        codemem_hard.emit_paraphrase(src, dest)

        before = {q["question_id"]: q for q in json.loads(src.read_text())}
        for q in json.loads(dest.read_text()):
            original = before[q["question_id"]]
            assert q["question"] == codemem_hard.PARAPHRASE_QUESTIONS[q["question_id"]]
            assert q["question"] != original["question"]
            assert q["original_question"] == original["question"]
            for key in ("haystack_session_ids", "answer_session_ids", "answer",
                        "question_type"):
                assert q[key] == original[key]

    def test_abstention_ids_map_through_their_suffix(self, tmp_path: Path) -> None:
        src = self._dataset(tmp_path, ["abs_redis_version_abs"])
        dest = tmp_path / "out.json"
        codemem_hard.emit_paraphrase(src, dest)
        (q,) = json.loads(dest.read_text())
        assert q["question_id"] == "abs_redis_version_abs"
        assert q["question"] == codemem_hard.PARAPHRASE_QUESTIONS["abs_redis_version"]

    def test_refuses_to_write_a_partial_split(self, tmp_path: Path) -> None:
        """Silently dropping the questions it has no paraphrase for would
        produce a smaller-n score presented as the full result."""
        src = self._dataset(tmp_path, ["arch_auth", "no_such_question"])
        dest = tmp_path / "out.json"
        with pytest.raises(SystemExit, match="no_such_question"):
            codemem_hard.emit_paraphrase(src, dest)
        assert not dest.exists()


# ----------------------------------------------------------------------
# Judge validation
# ----------------------------------------------------------------------

class _StubJudge:
    """Stands in for the model call so the rates can be checked exactly."""

    def __init__(self, verdict) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, str, str, str]] = []

    async def __call__(self, question, gold, candidate, model, **kw):
        self.calls.append((question, gold, candidate, model))
        accepted, explanation = self.verdict(candidate)
        return accepted, 1.0 if accepted else 0.0, explanation, 1.0


@pytest.fixture
def stub_judge(monkeypatch):
    """Install a stub in place of the real judge for the duration of a test."""
    def install(verdict):
        stub = _StubJudge(verdict)
        module = type(sys)("longmemeval_e2e_bench")
        module.judge_answer = stub
        monkeypatch.setitem(sys.modules, "longmemeval_e2e_bench", module)
        return stub
    return install


class TestRunJudge:
    def test_a_perfect_judge_scores_zero_on_both_error_rates(self, stub_judge) -> None:
        wrongs = {a["wrong"] for a in codemem_hard.ADVERSARIAL_ANSWERS}
        stub_judge(lambda c: (c not in wrongs, "CORRECT" if c not in wrongs else "INCORRECT"))

        out = codemem_hard.run_judge("stub/model")
        assert out["cells"]["adversarial"]["rate"] == 0.0
        assert out["cells"]["restated"]["rate"] == 0.0
        assert out["cells"]["verbatim"]["rate"] == 0.0

    def test_a_judge_that_accepts_everything_shows_a_full_false_accept_rate(
        self, stub_judge
    ) -> None:
        stub_judge(lambda c: (True, "CORRECT"))
        out = codemem_hard.run_judge("stub/model")
        assert out["cells"]["adversarial"]["rate"] == 1.0
        assert out["cells"]["restated"]["rate"] == 0.0

    def test_a_judge_that_rejects_everything_is_caught_by_the_controls(
        self, stub_judge
    ) -> None:
        """The reason the correct-answer cells exist. Rejecting everything
        scores a flawless 0% false-accept rate, and only the controls reveal
        that the judge is not measuring anything."""
        stub_judge(lambda c: (False, "INCORRECT"))
        out = codemem_hard.run_judge("stub/model")
        assert out["cells"]["adversarial"]["rate"] == 0.0, "looks perfect..."
        assert out["cells"]["restated"]["rate"] == 1.0, "...but the control exposes it"
        assert out["cells"]["verbatim"]["rate"] == 1.0

    def test_a_judge_grading_wording_fails_only_the_restated_cell(self, stub_judge) -> None:
        golds = {f["answer"] for f in codemem_hard._load_facts()}
        stub_judge(lambda c: (c in golds, "CORRECT" if c in golds else "INCORRECT"))
        out = codemem_hard.run_judge("stub/model")
        assert out["cells"]["verbatim"]["rate"] == 0.0
        assert out["cells"]["restated"]["rate"] == 1.0
        assert out["cells"]["adversarial"]["rate"] == 0.0

    def test_failed_calls_never_become_a_published_rate(self, stub_judge) -> None:
        """`judge_answer` scores an errored call as a rejection, so an expired
        credential would otherwise read as a flawless 0% false-accept rate."""
        stub_judge(lambda c: (False, "Judge error after 3 attempts: 401"))
        with pytest.raises(SystemExit, match="judge calls failed"):
            codemem_hard.run_judge("stub/model")

    def test_every_case_carries_a_question_and_a_gold_answer(self, stub_judge) -> None:
        stub = stub_judge(lambda c: (True, "CORRECT"))
        codemem_hard.run_judge("stub/model")
        assert len(stub.calls) == (len(codemem_hard.ADVERSARIAL_ANSWERS)
                                   + len(codemem_hard.EQUIVALENT_ANSWERS)
                                   + len(codemem_hard._load_facts()))
        for question, gold, candidate, model in stub.calls:
            assert question and gold and candidate
            assert model == "stub/model"

    def test_validation_cases_reference_real_facts(self) -> None:
        known = {f["id"] for f in codemem_hard._load_facts()}
        for case in codemem_hard.ADVERSARIAL_ANSWERS + codemem_hard.EQUIVALENT_ANSWERS:
            assert case["fact_id"] in known

    def test_adversarial_answers_are_not_accidentally_the_gold_answer(self) -> None:
        facts = {f["id"]: f for f in codemem_hard._load_facts()}
        for a in codemem_hard.ADVERSARIAL_ANSWERS:
            assert a["wrong"] != facts[a["fact_id"]]["answer"]
