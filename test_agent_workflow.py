import warnings
warnings.filterwarnings("ignore")

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import OllamaModel
from football_agent import create_football_agent

local_judge = OllamaModel(model="llama3:latest")
relevancy_metric = AnswerRelevancyMetric(threshold=0.8, model=local_judge)
faithfulness_metric = FaithfulnessMetric(threshold=0.9, model=local_judge)

def run_agent_and_capture(question):
    agent = create_football_agent()
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    final_answer = result["messages"][-1].content
    tool_calls_made = []
    for message in result["messages"]:
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls_made.append({
                    "tool": tc["name"],
                    "args": tc["args"]
                })
    return final_answer, tool_calls_made


def test_two_tool_workflow():
    """Agent must call both get_player_stats and search_match_reports"""
    question = "How many goals has Martinelli scored this season and did he score in the Arsenal vs Manchester City match?"

    actual_output, tool_calls = run_agent_and_capture(question)
    tools_called = [tc["tool"] for tc in tool_calls]

    assert "get_player_stats" in tools_called, \
        f"Expected get_player_stats but got: {tools_called}"
    assert "search_match_reports" in tools_called, \
        f"Expected search_match_reports but got: {tools_called}"
    assert "18" in actual_output, \
        f"Expected 18 goals in answer but got: {actual_output}"
    assert "23" in actual_output, \
        f"Expected 23rd minute goal in answer but got: {actual_output}"


def test_step_sequencing():
    """Agent must call tools and produce an answer that uses both results"""
    question = "Who is top of the Premier League and how many goals has their top scorer Haaland scored?"

    actual_output, tool_calls = run_agent_and_capture(question)
    tools_called = [tc["tool"] for tc in tool_calls]

    assert "get_league_table" in tools_called, \
        f"Expected get_league_table but got: {tools_called}"
    assert "get_player_stats" in tools_called, \
        f"Expected get_player_stats but got: {tools_called}"
    assert "Arsenal" in actual_output, \
        f"Expected Arsenal as league leaders in answer: {actual_output}"
    assert "32" in actual_output, \
        f"Expected Haaland's 32 goals in answer: {actual_output}"


def test_error_propagation():
    """Agent should handle no results gracefully without hallucinating"""
    question = "Who scored in the Arsenal vs Liverpool match?"

    actual_output, tool_calls = run_agent_and_capture(question)
    tools_called = [tc["tool"] for tc in tool_calls]

    assert "search_match_reports" in tools_called, \
        f"Expected search_match_reports but got: {tools_called}"
    assert "No match report found" in actual_output or \
           "couldn't find" in actual_output.lower() or \
           "don't have" in actual_output.lower() or \
           "not available" in actual_output.lower() or \
           "unable" in actual_output.lower() or \
           "no information" in actual_output.lower(), \
        f"Agent hallucinated data for a match not in database: {actual_output}"