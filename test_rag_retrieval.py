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

test_case_correct_retrieval = LLMTestCase(
    input="Who are The Arsenal?",
    actual_output="Arsenal Football Club is a North London based professional football club",
    retrieval_context=["The Arsenal Football Club is a professional football club based in Islington, North London, England"]
)

test_case_wrong_document = LLMTestCase(
    input="Who are the Arsenal?",
    actual_output="Chelsea Football Club is a West London based professional football club",
    retrieval_context=["Chelsea Football Club is a professional football club based in Fulham, West London, England"]
)

test_case_incomplete_retrieval = LLMTestCase(
    input="Who is Bukayo Saka?",
    actual_output="Bukayo Saka is an English professional footballer who plays for Arsenal",
    retrieval_context=["Bukayo Saka (born 5 September 2001) is an English professional footballer who plays as a right winger for Premier League club Arsenal "]
)

test_case_no_retrieval = LLMTestCase(
    input="Who is Mikel Arteta?",
    actual_output="Mikel Arteta is the manager of Premier League club Arsenal",
    retrieval_context=[]
)

test_case_partially_relevant = LLMTestCase(
    input="Who is Robin Van Persie",
    actual_output="Robin Van Persie eat rats",
    retrieval_context=[
        "Robin Van Persie is a Dutch former professional footballer who played for Arsenal",
        "Arsenal vs Chelsea match report: Saka scored in the 45th minute"
    ]
)

def test_correct_retrieval():
    assert_test(test_case_correct_retrieval, [relevancy_metric, faithfulness_metric, context_relevancy_metric])

def test_wrong_document():
    assert_test(test_case_wrong_document, [relevancy_metric, faithfulness_metric, context_relevancy_metric])

def test_incomplete_retrieval():
    assert_test(test_case_incomplete_retrieval, [relevancy_metric, faithfulness_metric, context_relevancy_metric])

def test_no_retrieval():
    assert_test(test_case_no_retrieval, [relevancy_metric, faithfulness_metric, context_relevancy_metric])

def test_partially_relevant():
    assert_test(test_case_partially_relevant, [relevancy_metric, faithfulness_metric, context_relevancy_metric])