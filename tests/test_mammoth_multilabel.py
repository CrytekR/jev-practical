import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / "examples" / "mammoth_multilabel.py"
_SPEC = importlib.util.spec_from_file_location("mammoth_multilabel", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
mammoth = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mammoth
_SPEC.loader.exec_module(mammoth)


def test_every_question_shares_the_state_and_keeps_its_own_suffix() -> None:
    rendered = [mammoth.render_question(qid, question) for qid, question in mammoth.QUESTIONS.items()]
    shared = mammoth.state_block(mammoth.TICKET)
    users = [mammoth.messages(mammoth.TICKET, item.suffix)[1]["content"] for item in rendered]
    assert all(content.startswith(shared) for content in users)
    assert len({content.removeprefix(shared) for content in users}) == len(rendered)
    refund, duplicate, team, urgency = rendered
    assert "charged twice" not in refund.suffix
    assert "duplicate" not in team.suffix
    assert refund.labels == ("yes", "no")
    assert duplicate.labels == ("yes", "no")
    assert team.labels == ("billing", "technical", "fraud")
    assert urgency.suffix.startswith("Rate the state:")
    assert "A. level 0: Can wait" in urgency.suffix
    assert "B. billing" not in urgency.suffix


def test_slot_softmax_is_the_answer_distribution() -> None:
    refund = mammoth.render_question("refund", mammoth.QUESTIONS["refund"])
    answer = mammoth.assemble(refund, [0.0, 0.0])
    assert answer["noul"] == pytest.approx(0.5)

    peaked = mammoth.assemble(refund, [4.0, 0.0])
    assert peaked["noul"] > 0.9

    team = mammoth.render_question("team", mammoth.QUESTIONS["team"])
    choice = mammoth.assemble(team, [0.2, 2.0, 0.2])
    assert choice["choice"] == "technical"
    assert sum(choice["probabilities"].values()) == pytest.approx(1.0)
    assert choice["confidence"] == pytest.approx(mammoth.choice_confidence(list(choice["probabilities"].values())))

    urgency = mammoth.render_question("urgency", mammoth.QUESTIONS["urgency"])
    score = mammoth.assemble(urgency, [0.0, 0.0, 0.0])
    assert score["score"] == pytest.approx(1.0)
    assert score["confidence"] == pytest.approx(0.0)


def test_tied_choice_keeps_the_earlier_slot() -> None:
    team = mammoth.render_question("team", mammoth.QUESTIONS["team"])
    assert mammoth.assemble(team, [1.0, 1.0, 0.0])["choice"] == "billing"


def test_explain_shows_one_state_and_four_suffixes() -> None:
    text = mammoth.explain(mammoth.TICKET, mammoth.QUESTIONS)
    assert text.count("<state>") == 1
    assert text.count("## ") == 4
    assert "不采样" in text
