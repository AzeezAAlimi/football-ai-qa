import os
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

IS_CI = os.getenv("CI") == "true"

GOLDEN_DATASET = [
    {
        "id": "TC001",
        "category": "match_results",
        "test_case": LLMTestCase(
            input="What was the final score in the Premier League match between Arsenal and Manchester City?",
            actual_output="The final score between Arsenal vs Manchester City ended in Arsenal winning 2-1",
            expected_output="The final score between Arsenal vs Manchester City is 2-1",
            retrieval_context=["Final score: Arsenal 2 Manchester City 1"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    {
        "id": "TC002",
        "category": "match_results",
        "test_case": LLMTestCase(
            input="Who scored in the game between Arsenal and Manchester City?",
            actual_output=" Martinelli and Odegaard scored for Arsenal and Haaland scored for Manchester City",
            expected_output="The goal scorers are Martinelli and Odegaard who scored for Arsenal and Haaland who scored for Manchester City",
            retrieval_context=["Scorers: Martinelli 23', Odegaard 67' (Arsenal) Haaland 45' (Manchester City)"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    {
        "id": "TC003",
        "category": "match_results",
        "test_case": LLMTestCase(
            input="Who refereed the game between Arsenal vs Manchester City?",
            actual_output="The referee for the Arsenal vs Manchester City match was Michael Oliver.",
            expected_output="The referee was Michael Oliver",
            retrieval_context=["Referee: Michael Oliver"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    {
        "id": "TC004",
        "category": "match_results",
        "test_case": LLMTestCase(
            input="Can you name the venue where the Arsenal vs Manchester City game was played?",
            actual_output="The Emirates Stadium",
            expected_output="The Emirates Stadium",
            retrieval_context=["Venue: Emirates Stadium, London"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    {
        "id": "TC005",
        "category": "match_results",
        "test_case": LLMTestCase(
            input="What was the attendance for the game between Arsenal and Manchester City?",
            actual_output="60,000",
            expected_output="The recorded attendance was 60,000",
            retrieval_context=["Attendance: 60,000"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    {
        "id": "TC006",
        "category": "player_performance",
        "test_case": LLMTestCase(
            input="How many shots on target did Saka have?",
            actual_output="Saka had 2 shots on target",
            expected_output="Saka had 2 shots on target",
            retrieval_context=["Saka shots on target: 2"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    {
        "id": "TC007",
        "category": "player_performance",
        "test_case": LLMTestCase(
            input="How many successful tackles did Declan Rice make?",
            actual_output="Rice made 5 successful tackles",
            expected_output="Rice made 5 successful tackles",
            retrieval_context=["Rice successful tackles: 5"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    {
        "id": "TC008",
        "category": "player_performance",
        "test_case": LLMTestCase(
            input="How many accurate crosses did Martinelli make?",
            actual_output="Martinelli made 3 accurate crosses",
            expected_output="Martinelli made 3 accurate crosses",
            retrieval_context=["Martinelli accurate crosses: 3"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    {
        "id": "TC009",
        "category": "player_performance",
        "test_case": LLMTestCase(
            input="How many dribbles did Trossard complete?",
            actual_output="Trossard completed 6 dribbles",
            expected_output="Trossard compelted 6 dribbles",
            retrieval_context=["Trossard dribble completed: 6"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    {
        "id": "TC010",
        "category": "player_performance",
        "test_case": LLMTestCase(
            input="How many aerial duels did Saliba win?",
            actual_output="Saliba won 9 aerial duels",
            expected_output="Saliba won 9 aerial duels",
            retrieval_context=["Saliba aerial duels: 9"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    {
        "id": "TC011",
        "category": "team_performance",
        "test_case": LLMTestCase(
            input="What was Arsenal’s possession percentage?",
            actual_output="Arsenal possession percentage is 48%",
            expected_output="Arsenal possession percentage is 48%",
            retrieval_context=["Arsenal possession: 48%"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    {
        "id": "TC012",
        "category": "team_performance",
        "test_case": LLMTestCase(
            input="What was Manchester City’s possession percentage?",
            actual_output="Manchester City possession percentage is 52%",
            expected_output="Manchester City possession percentage is 52%",
            retrieval_context=["Manchester City possession: 52%"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    {
        "id": "TC013",
        "category": "team_performance",
        "test_case": LLMTestCase(
            input="How many shots on target did Arsenal have?",
            actual_output="Arsenal had 6 shots on target",
            expected_output="Arsenal had 6 shots on target",
            retrieval_context=["Arsenal shots on target: 6"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    {
        "id": "TC014",
        "category": "team_performance",
        "test_case": LLMTestCase(
            input="How many shots on target did Manchester City have?",
            actual_output="Manchester City had 9 shots on target",
            expected_output="Manchester City had 9 shots on target",
            retrieval_context=["Manchester City shots on target: 9"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    {
        "id": "TC015",
        "category": "team_performance",
        "test_case": LLMTestCase(
            input="What was Arsenal’s pass accuracy?",
            actual_output="Arsenal pass accuracy is 84%",
            expected_output="Arsenal pass accuracy is 84%",
            retrieval_context=["Arsenal pass accuracy: 84%"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    # TC016 — Ambiguous question (model uses match report to clarify)
    {
        "id": "TC016",
        "category": "match_results",
        "test_case": LLMTestCase(
            input="Who was booked in the match?",
            actual_output="Rodri from Manchester City was booked in the 34th minute.",
            expected_output="Rodri from Manchester City received a booking.",
            retrieval_context=["Bookings: Rodri (Manchester City) 34'"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    # TC017 — Missing information (model acknowledges correctly)
    {
        "id": "TC017",
        "category": "match_results",
        "test_case": LLMTestCase(
            input="Who was named man of the match?",
            actual_output="Martin Odegaard was named man of the match.",
            expected_output="Martin Odegaard was man of the match.",
            retrieval_context=["Man of the match: Martin Odegaard"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    # TC018 — Wrong assumption (model corrects it)
    {
        "id": "TC018",
        "category": "team_performance",
        "test_case": LLMTestCase(
            input="What was Manchester City's pass accuracy?",
            actual_output="Manchester City's pass accuracy was 89%.",
            expected_output="Manchester City had a pass accuracy of 89%.",
            retrieval_context=["Manchester City pass accuracy: 89%"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    },
    # TC019 — Partial information (model answers what it can)
    {
    "id": "TC019",
    "category": "match_results",
    "test_case": LLMTestCase(
        input="When was the Arsenal vs Manchester City match played?",
        actual_output="The match was played on 12 April 2026.",
        expected_output="The match was played on 12 April 2026.",
        retrieval_context=["Date: 12 April 2026"]
    ),
    "min_relevancy": 0.8,
    "min_faithfulness": 0.9
    },
    {
        "id": "TC020",
        "category": "team_performance",
        "test_case": LLMTestCase(
            input="How many shots on target did each team have?",
            actual_output="Arsenal had 6 shots on target and Manchester City had 9 shots on target.",
            expected_output="Arsenal had 6 and Manchester City had 9 shots on target.",
            retrieval_context=["Arsenal shots on target: 6", "Manchester City shots on target: 9"]
        ),
        "min_relevancy": 0.8,
        "min_faithfulness": 0.9
    }
]

# --- TEST VARIANT 1: Runs locally with Ollama, skips in CI ---
@pytest.mark.skipif(IS_CI, reason="Option B: Skipping full LLM metrics in Github Actions")
@pytest.mark.parametrize("entry", GOLDEN_DATASET, ids=lambda x: x["id"])
def test_golden_dataset_local(entry):
    from deepeval.models import OllamaModel

    local_judge = OllamaModel(model="llama3:latest")

    relevancy_metric = AnswerRelevancyMetric(threshold=entry["min_relevancy"], model=local_judge)
    faithfulness_metric = FaithfulnessMetric(threshold=entry["min_faithfulness"], model=local_judge)

    assert_test(entry["test_case"], [relevancy_metric, faithfulness_metric])

# --- TEST VARIANT 2: Smoke check that executes everywhere (including GitHub CI) ---
def test_golden_dataset_smoke():
    """
    Smoke test — runs in CI and locally.
    Validates dataset structure without LLM evaluation.
    """
    assert len(GOLDEN_DATASET) == 20, "Golden dataset must have 20 test cases"

    for entry in GOLDEN_DATASET:
        assert "id" in entry, f"Missing id in entry"
        assert "category" in entry, f"Missing category in {entry.get('id')}"
        assert "test_case" in entry, f"Missing test_case in {entry.get('id')}"
        assert "min_relevancy" in entry, f"Missing min_relevancy in {entry.get('id')}"
        assert "min_faithfulness" in entry, f"Missing min_faithfulness in {entry.get('id')}"
        assert entry["min_relevancy"] >= 0.8, f"{entry['id']}: relevancy threshold too low"
        assert entry["min_faithfulness"] >= 0.9, f"{entry['id']}: faithfulness threshold too low"

    print(f"✅ Smoke test passed — {len(GOLDEN_DATASET)} test cases validated")

def test_quality_gate_thresholds():
    """
    Validates that all quality gate thresholds
    meet minimum enterprise standards.
    Fails if any threshold is set too low.
    """
    MINIMUM_RELEVANCY_THRESHOLD = 0.8
    MINIMUM_FAITHFULNESS_THRESHOLD = 0.9

    for entry in GOLDEN_DATASET:
        assert entry["min_relevancy"] >= MINIMUM_RELEVANCY_THRESHOLD, \
            f"{entry['id']}: relevancy threshold {entry['min_relevancy' ]} below minimum {MINIMUM_RELEVANCY_THRESHOLD}"
        assert entry["min_faithfulness"] >= MINIMUM_FAITHFULNESS_THRESHOLD, \
            f"{entry['id']}: faithfulness threshold {entry['min_faithfulness']} below minimum {MINIMUM_FAITHFULNESS_THRESHOLD}"

    print(f"✅ Quality gate thresholds validated across {len(GOLDEN_DATASET)} test cases")