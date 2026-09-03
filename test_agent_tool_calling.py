import warnings
warnings.filterwarnings("ignore")

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import OllamaModel
from football_agent import create_football_agent

# Set up the judge
local_judge = OllamaModel(model="llama3:latest")
relevancy_metric = AnswerRelevancyMetric(threshold=0.8, model=local_judge)
faithfulness_metric = FaithfulnessMetric(threshold=0.9, model=local_judge)

# Helper — runs the agent and captures what it did
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


def test_hallucination_detection():
    question = "Who scored in the Arsenal vs Manchester City match?"

    actual_output, tool_calls = run_agent_and_capture(question)

    # Assert Eze is NOT in the answer — he didn't play
    assert "Eze" not in actual_output, \
        f"Hallucination detected — Eze in answer: {actual_output}"

    # Assert Odegaard and Haaland ARE in the answer
    assert "Odegaard" in actual_output, \
        f"Odegaard missing: {actual_output}"
    assert "Haaland" in actual_output, \
        f"Haaland missing: {actual_output}"


def test_correct_tool_selection():
    question = "Who scored in the Arsenal vs Manchester City match?"

    actual_output, tool_calls = run_agent_and_capture(question)

    # Get list of tools that were called
    tools_called = [tc["tool"] for tc in tool_calls]

    # Assert the right tool was called
    assert "search_match_reports" in tools_called, \
        f"Expected search_match_reports but agent called: {tools_called}"

    # Assert the wrong tools were NOT called
    assert "get_player_stats" not in tools_called, \
        f"Agent called wrong tool get_player_stats for a match question"

    assert "get_league_table" not in tools_called, \
        f"Agent called wrong tool get_league_table for a match question"


def test_player_stats_tool():
    question = "How many goals has Saka scored this season?"

    actual_output, tool_calls = run_agent_and_capture(question)

    # Assert correct tool was called
    tools_called = [tc["tool"] for tc in tool_calls]
    assert "get_player_stats" in tools_called, \
        f"Expected get_player_stats but agent called: {tools_called}"

    # Assert correct stats in answer
    assert "22" in actual_output, \
        f"Expected 22 goals in answer but got: {actual_output}"


def test_league_table_tool():
    question = "Where does Arsenal sit in the league table?"

    actual_output, tool_calls = run_agent_and_capture(question)

    # Assert correct tool was called
    tools_called = [tc["tool"] for tc in tool_calls]
    assert "get_league_table" in tools_called, \
        f"Expected get_league_table but agent called: {tools_called}"

    # Assert Arsenal are 1st with 82 points
    assert "82" in actual_output, \
        f"Expected 82 points in answer but got: {actual_output}"


def test_out_of_scope_question():
    question = "What is the weather in London today?"

    actual_output, tool_calls = run_agent_and_capture(question)

    # Assert the agent didn't call any football tools
    tools_called = [tc["tool"] for tc in tool_calls]
    assert "search_match_reports" not in tools_called, \
        f"Agent incorrectly called search_match_reports for a weather question"

    # Assert the agent didn't hallucinate weather data
    assert "celsius" not in actual_output.lower(), \
        f"Agent hallucinated temperature data: {actual_output}"
    assert "sunny" not in actual_output.lower(), \
        f"Agent hallucinated weather data: {actual_output}"