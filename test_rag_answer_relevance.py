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

test_case_perfect_response = LLMTestCase(
    input="What arsenal player won the man of the match?",
    actual_output="The man of the match is Martin Odegaard",
    retrieval_context=["Man of the match: Martin Odegaard (Arsenal)"]
)

test_case_answer_is_faithful_but_off_topic = LLMTestCase(
    input="What Manchester City player got booked?",
    actual_output="Rodri from Manchester City received a booking in the 34th minute",
    retrieval_context=["Bookings: Rodri (Manchester City) 34'"]
)

test_case_answer_is_relevant_but_unfaithful = LLMTestCase(
    input="Who refereed the game?",
    actual_output="Michael Oliver who is hunger",
    retrieval_context=["Referee: Michael Oliver"]
)

test_case_answer_is_vague_and_padded = LLMTestCase(
    input="How many shots did Arsenal have on target?",
    actual_output="This number is in between 5 and 7",
    retrieval_context=["Arsenal shots on target: 6"]
)

test_case_answer_contradicts_the_question_premise = LLMTestCase(
    input="What was the match attendance?",
    actual_output="President Trump is the current American President",
    retrieval_context=["Attendance: 60,000"]
)

def test_perfect_response():
    assert_test(test_case_perfect_response, [relevancy_metric, faithfulness_metric, context_relevancy_metric])

def test_answer_is_faithful_but_off_topic():
    assert_test(test_case_answer_is_faithful_but_off_topic, [relevancy_metric, faithfulness_metric, context_relevancy_metric])

def test_answer_is_relevant_but_unfaithful():
    assert_test(test_case_answer_is_relevant_but_unfaithful, [relevancy_metric, faithfulness_metric, context_relevancy_metric])

def test_answer_is_vague_and_padded():
    assert_test(test_case_answer_is_vague_and_padded, [relevancy_metric, faithfulness_metric, context_relevancy_metric])

def test_answer_contradicts_the_question_premise():
    assert_test(test_case_answer_contradicts_the_question_premise, [relevancy_metric, faithfulness_metric, context_relevancy_metric])