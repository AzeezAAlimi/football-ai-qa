nst# RAG Test Strategy — Football AI Assistant

## System Overview
This project tests a RAG-based football AI assistant that answers 
questions about match reports. The system retrieves relevant match 
report chunks from a database and generates grounded answers using 
an LLM.

## Components Under Test
- **Retriever** — searches the match report database for relevant chunks
- **Context Window** — the retrieved chunks injected alongside the question
- **Generator** — the LLM that produces the final answer

## Test Approach
- Framework: DeepEval
- Judge model: llama3:latest (local via Ollama)
- Metrics: AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
- Test runner: pytest + GitHub Actions CI/CD

## Test Coverage

| File | What it tests | Test cases |
|---|---|---|
| test_rag_retrieval.py | Retrieval quality | 5 |
| test_rag_answer_relevance.py | Answer relevance failures | 5 |
| test_rag_faithfulness.py | Faithfulness failures | 5 |
| test_rag_pipeline.py | Full end-to-end pipeline | 5 |

## Quality Gates

| Metric | Threshold |
|---|---|
| AnswerRelevancyMetric | >= 0.8 |
| FaithfulnessMetric | >= 0.9 |
| ContextualRelevancyMetric | >= 0.7 |
| Pass rate | 100% |

## Known Limitations
- Judge model is qwen3:14b (local) — a stronger judge like GPT-4 
  would produce more reliable scores
- Edge case refusals are not reliably evaluated by llama3
- No live retriever connected — retrieval_context is manually 
  provided to simulate retrieval

## Recommendations
- Upgrade to GPT-4 as judge model for production
- Connect a real vector database retriever
- Add adversarial test cases for prompt injection
- Increase golden dataset to 50+ test cases