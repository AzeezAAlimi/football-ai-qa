from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.models import OllamaModel

local_judge = OllamaModel(model="qwen3:14b")

relevancy_metric = AnswerRelevancyMetric(
    threshold=0.8,
    model=local_judge
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.9,
    model=local_judge
)

test_case_perfect_faithfulness = LLMTestCase(
    input="What was the final score between Arsenal vs Manchester City?",
    actual_output="The final score was Arsenal 2 Manchester City 1",
    expected_output="The score was Arsenal 2 Manchester City 1",
    retrieval_context=["Final score: Arsenal 2 Manchester City 1"]
)

test_case_training_data_bleed = LLMTestCase(
    input="What was the final score between Arsenal vs Manchester City?",
    actual_output="The final score was Arsenal 2 Manchester City 1 and Arsenal are now top of the league",
    expected_output="The score was Arsenal 2 Manchester City 1",
    retrieval_context=["Final score: Arsenal 2 Manchester City 1"]
)

test_case_context_interpolation = LLMTestCase(
    input="Who scored in the Arsenal vs Manchester City match?",
    actual_output="Martinelli scored an overhead kick on the 23' min and then Odegaard scored a header on the 67' minute",
    expected_output="Martinelli scored on the 23' min and then Odegaard scored on the 67' min",
    retrieval_context=["Scorers: Martinelli 23', Odegaard 67' (Arsenal)"]
)

test_case_confident_extrapolation = LLMTestCase(
    input="Who got booking during the match?",
    actual_output="Rodri got booked on the 34' min and had to get held back by his team mates",
    expected_output="Rodri got booked on the 34' min",
    retrieval_context=["Bookings: Rodri (Manchester City) 34'"]
)

test_case_complete_hallucination = LLMTestCase(
    input="Who refereed the game?",
    actual_output="Usain Bolt is fast",
    expected_output="Michael Oliver",
    retrieval_context=["Referee: Michael Oliver"]
)

def test_perfect_faithfulness():
    assert_test(test_case_perfect_faithfulness, [relevancy_metric, faithfulness_metric])

def test_training_data_bleed():
    assert_test(test_case_training_data_bleed, [relevancy_metric, faithfulness_metric])

def test_context_interpolation():
    assert_test(test_case_context_interpolation, [relevancy_metric, faithfulness_metric])

def test_confident_extrapolation():
    assert_test(test_case_confident_extrapolation, [relevancy_metric, faithfulness_metric])

def test_complete_hallucination():
    assert_test(test_case_complete_hallucination, [relevancy_metric, faithfulness_metric])