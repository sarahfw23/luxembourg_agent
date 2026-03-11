# Prediction Prophet GPT-4o: How It Works

The **Prediction Prophet GPT-4o** agent is a sophisticated multi-agent system designed to participate in prediction markets (like Omen or PolyMarket) by performing deep research and making probabilistic forecasts.

## 🏗️ Core Architecture

The agent follows a **Dual-Agent Architecture** powered by OpenAI's `gpt-4o-2024-08-06` model. It separates "finding truth" from "making a prediction."

### 1. Research Agent
*   **Role**: Information gathering and synthesis.
*   **Model Config**: `gpt-4o` with **Temperature 0.7**. Higher temperature allows for more creative and diverse search queries to ensure broad coverage.
*   **Workflow**:
    *   **Subquery Generation**: Breaks down the market question into 3-5 specific subqueries.
    *   **Web Search**: Uses **Tavily** or **Google** to find contemporary news and data.
    *   **Scraping**: Scrapes full text from relevant websites (filtering out noise and non-relevant sites like YouTube).
    *   **Context Retrieval**: Uses a vector database (with OpenAI embeddings) to find the most relevant "snippets" of info for the specific question.
    *   **Synthesis**: Combines all findings into a structured **Research Report**.

### 2. Prediction Agent
*   **Role**: Final forecasting and reasoning.
*   **Model Config**: `gpt-4o` with **Temperature 0.0**. Zero temperature ensures deterministic, logical, and precise probability estimations.
*   **Workflow**:
    *   Takes the **Market Question** + **Research Report**.
    *   Follows structured prompts (`PREDICTION_PROMPT`) that emphasize:
        *   **Recency**: Newer information overrides older data.
        *   **Closing Dates**: Analyzing if an event will happen *before* or *after* the market settles.
        *   **Isolation**: Ignoring the market creator's intent and focusing on real-world protagonists.
    *   Outputs a **Probabilistic Answer** (e.g., `Yes: 75%, No: 25%`) along with detailed reasoning.

---

## 💰 Betting Strategy: The Kelly Criterion

The agent doesn't just predict; it decides how much to bet using the **Kelly Criterion**.

*   **Logic**: It compares its predicted probability ($P_{agent}$) with the current market price ($P_{market}$).
*   **Risk Management**: It calculates an "edge" and bets a fraction of its balance proportional to that edge.
*   **Safety Features**:
    *   **Max Price Impact**: Won't bet if the trade would move the market price too much.
    *   **Take Profit**: Has logic to close positions early if a target profit is reached (configurable).

---

## 🛠️ Implementation Details

*   **Tooling**: Built on top of `pydantic-ai` and `prediction-market-agent-tooling`.
*   **Prompting**: Uses a JSON-based output format for seamless integration with trading contracts.
*   **Environment**: Typically deployed using a `DeployableTraderAgent` wrapper which handles blockchain interactions and trade intervals.

### Example Workflow:
1.  **Question**: "Will Bitcoin hit $100k by March 2026?"
2.  **Research**: Find BTC ETFs inflows, halving cycles, and latest FED sentiment.
3.  **Synthesis**: Report says "90% of analysts expect $120k by end of 2025."
4.  **Prediction**: Agent calculates a 92% probability.
5.  **Bet**: If market price is $0.85 (85%), agent identifies a 7% edge and places a Kelly-sized bet on `Yes`.
