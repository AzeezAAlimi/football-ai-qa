from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import warnings
warnings.filterwarnings("ignore")


# ── Tools ──────────────────────────────────────────────────────────────────

@tool
def search_match_reports(query: str) -> str:
    """Search for football match reports. Use this when asked about match results, scores, scorers, attendance, or referee."""
    match_data = {
        "arsenal manchester city": """
            Arsenal vs Manchester City - Premier League
            Final score: Arsenal 2 Manchester City 1
            Scorers: Martinelli 23', Odegaard 67' (Arsenal), Haaland 45' (Manchester City)
            Attendance: 60,000
            Referee: Michael Oliver
            Arsenal possession: 48%, Manchester City possession: 52%
            Man of the match: Martin Odegaard
            Bookings: Rodri (Manchester City) 34'
        """,
        "arsenal chelsea": """
            Arsenal vs Chelsea - Premier League
            Final score: Arsenal 3 Chelsea 1
            Scorers: Saka 12', 45' (Arsenal), Martinelli 78' (Arsenal), Palmer 60' (Chelsea)
            Attendance: 60,000
            Referee: Anthony Taylor
            Man of the match: Bukayo Saka
        """
    }

    query_lower = query.lower()
    for key, report in match_data.items():
        if all(word in query_lower for word in key.split()):
            return report

    return "No match report found for that query."


@tool
def get_player_stats(player_name: str) -> str:
    """Get player statistics for the current season. Use this when asked about a player's goals, assists, or appearances."""
    stats = {
        "saka": "Bukayo Saka — Arsenal — Season stats: 22 goals, 14 assists, 35 appearances",
        "martinelli": "Gabriel Martinelli — Arsenal — Season stats: 18 goals, 8 assists, 33 appearances",
        "odegaard": "Martin Odegaard — Arsenal — Season stats: 15 goals, 12 assists, 34 appearances",
        "haaland": "Erling Haaland — Manchester City — Season stats: 32 goals, 6 assists, 36 appearances",
        "rice": "Declan Rice — Arsenal — Season stats: 8 goals, 10 assists, 35 appearances"
    }

    name_lower = player_name.lower()
    for key, stat in stats.items():
        if key in name_lower:
            return stat

    return f"No stats found for {player_name}."


@tool
def get_league_table() -> str:
    """Get the current Premier League standings. Use this when asked about league position, points, or standings."""
    return """
        Premier League Standings (Top 5):
        1. Arsenal — 82 points — 35 played — GD +48
        2. Manchester City — 78 points — 35 played — GD +44
        3. Liverpool — 74 points — 35 played — GD +39
        4. Chelsea — 62 points — 35 played — GD +18
        5. Aston Villa — 58 points — 35 played — GD +12
    """


# ── Agent setup ─────────────────────────────────────────────────────────────

def create_football_agent():
    llm = ChatOllama(model="qwen3:14b", temperature=0)
    tools = [search_match_reports, get_player_stats, get_league_table]
    return create_react_agent(llm, tools)


# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = create_football_agent()

    questions = [
        "Who scored in the Arsenal vs Manchester City match?",
        "How many goals has Saka scored this season?",
        "Where does Arsenal sit in the league table?"
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        final_message = result["messages"][-1]
        print(f"\nFinal Answer: {final_message.content}")