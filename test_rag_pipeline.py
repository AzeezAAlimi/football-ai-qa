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

context_relevancy_metric = ContextualRelevancyMetric(
    threshold=0.7,
    model=local_judge
)

test_case_1 = LLMTestCase(
    input="Who refereed the game between Arsenal vs Manchester City?",
    actual_output="The referee is Michael Oliver",
    expected_output="Michael Oliver is the referee",
    retrieval_context=["Referee: Michael Oliver, Arsenal vs Manchester City - Premier League"]
)

test_case_2 = LLMTestCase(
    input="Who is the man of the match from the Arsenal vs Manchester City game?",
    actual_output="The man of the match is Martin Odegaard ",
    expected_output="Martin Odegaard is the man of the match",
    retrieval_context=["Man of the match: Martin Odegaard"]
)

test_case_3 = LLMTestCase(
    input="What was the match attendance?",
    actual_output="The match attendance is 60,000",
    expected_output="The match attendance is approximately 60,000",
    retrieval_context=["Attendance: 60,000"]
)

test_case_4 = LLMTestCase(
    input="What is the final score between Arsenal vs Manchester City?",
    actual_output="The final score is Arsenal 2 Manchester City 1",
    expected_output="The score is Arsenal 2 Manchester City 1",
    retrieval_context=["Final score: Arsenal 2 Manchester City 1"]
)

test_case_5 = LLMTestCase(
    input="What Manchester City football player got booked?",
    actual_output="Rodri from Manchester City was booked in the 34th minute",
    expected_output="Rodri from Manchester City was booked",
    retrieval_context=["Bookings: Rodri (Manchester City) 34'"]
)

def test_1():
    assert_test(test_case_1, [relevancy_metric, faithfulness_metric, context_relevancy_metric])

def test_2():
    assert_test(test_case_2, [relevancy_metric, faithfulness_metric, context_relevancy_metric])

def test_3():
    assert_test(test_case_3, [relevancy_metric, faithfulness_metric, context_relevancy_metric])

def test_4():
    assert_test(test_case_4, [relevancy_metric, faithfulness_metric, context_relevancy_metric])

def test_5():
    assert_test(test_case_5, [relevancy_metric, faithfulness_metric, context_relevancy_metric])