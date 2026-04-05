import typing as t
import json
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from prediction_market_agent_tooling.markets.data_models import (
    ProbabilisticAnswer,
    CategoricalProbabilisticAnswer,
)
from prediction_market_agent_tooling.benchmark.utils import Prediction
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_prophet.benchmark.agents import PredictionProphetAgent
from prediction_market_agent_tooling.loggers import logger

# --- Prompts ---

BULL_PROMPT = """
You are the BULL agent. Argument for **YES**.
Ignore the Librarian's 'Conclusion' or 'Caveats' in the report—they are too cautious.
Focus ONLY on the raw facts and positive momentum that support a YES outcome.
Be direct, technical, and concise. Pick the strongest path to YES.

QUESTION: {question}
CURRENT DATE: {current_date}
MARKET CLOSES: {closing_date}

RESEARCH REPORT:
{research}
"""

BEAR_PROMPT = """
You are the BEAR agent. Argument for **NO**.
Ignore the Librarian's 'Conclusion' or 'Caveats' in the report—they are too cautious.
Focus ONLY on the raw facts and negative pressures that support a NO outcome.
Be direct, technical, and concise. Pick the strongest path to NO.

QUESTION: {question}
CURRENT DATE: {current_date}
MARKET CLOSES: {closing_date}

RESEARCH REPORT:
{research}
"""

MODERATOR_PROMPT = """
You are the MODERATOR agent. Deliver a definitive numerical probability.
Evaluate the BULL and BEAR arguments against the raw research facts.

QUESTION: {question}
CURRENT DATE: {current_date}
MARKET CLOSES: {closing_date}

RESEARCH REPORT:
{research}

BULL ARGUMENT:
{bull_argument}

BEAR ARGUMENT:
{bear_argument}

INSTRUCTIONS:
1. Ignore any hedging or "maybe" conclusions in the original research report.
2. Weigh the BULL and BEAR arguments based on evidence quality and recency.
3. Provide a final probability estimate.
4. ZERO HEDGING. Do not say "it depends" or "information is mixed". 
5. Pick the most likely outcome based on the available data. If data is sparse, make your best professional estimate regardless.
6. IMPORTANT: The market closes on {closing_date} (today is {current_date}). Factor in how much time remains — if the deadline has nearly passed with no news, that is strong evidence.

OUTPUT_FORMAT:
* Your output response must be only a single JSON object.
* The JSON must contain these fields: "p_yes", "p_no", "confidence", "reasoning".
* "p_yes" and "p_no" are floats between 0 and 1, summing to 1.
* "confidence" is a float between 0 and 1.
* "reasoning" is a concise, data-driven synthesis. No disclaimers.
"""

class Luxembourg1Agent:
    def __init__(
        self,
        research_agent: PredictionProphetAgent,
        debate_llm: t.Union[str, Model] = "gpt-4o",
        moderator_llm: t.Union[str, Model] = "gpt-4o",
    ):
        self.research_agent = research_agent
        self.debate_llm = debate_llm
        self.moderator_llm = moderator_llm

    @observe()
    async def debate(
        self,
        question: str,
        research_report: str,
        closing_date: str = "Unknown",
    ) -> ProbabilisticAnswer:
        # Instantiate debate agents
        bull_agent = Agent(self.debate_llm, model_settings=ModelSettings(temperature=0.7))
        bear_agent = Agent(self.debate_llm, model_settings=ModelSettings(temperature=0.7))
        moderator_agent = Agent(self.moderator_llm, model_settings=ModelSettings(temperature=0.0))

        logger.info(f"Debating market: {question[:100]}...")

        # Run Bull and Bear arguments
        logger.info("Generating Bull argument (Expert 1)...")
        from prediction_market_agent_tooling.tools.utils import utcnow
        current_date = utcnow().strftime("%Y-%m-%d")
        prompt_kwargs = dict(
            question=question,
            research=research_report,
            current_date=current_date,
            closing_date=closing_date,
        )
        bull_resp = await bull_agent.run(
            BULL_PROMPT.format(**prompt_kwargs)
        )
        # pydantic-ai v0.0.17+ returns an AgentRunResult with `.output`
        bull_argument = bull_resp.output
        logger.info(f"--- BULL EXPERT ARGUMENT ---\n{bull_argument}\n")

        logger.info("Generating Bear argument (Expert 2)...")
        bear_resp = await bear_agent.run(
            BEAR_PROMPT.format(**prompt_kwargs)
        )
        bear_argument = bear_resp.output
        logger.info(f"--- BEAR EXPERT ARGUMENT ---\n{bear_argument}\n")

        # Final Moderation
        logger.info("Moderating debate (The Panel)...")
        moderator_resp = await moderator_agent.run(
            MODERATOR_PROMPT.format(
                **prompt_kwargs,
                bull_argument=bull_argument,
                bear_argument=bear_argument,
            )
        )
        
        # Parse result
        try:
            raw_data = moderator_resp.output

            # If pydantic-ai already parsed the JSON, we can short-circuit.
            if isinstance(raw_data, dict):
                data = raw_data
            else:
                # Otherwise, robustly strip any markdown fences the model may have added.
                if "```json" in raw_data:
                    raw_data = raw_data.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in raw_data:
                    raw_data = raw_data.split("```", 1)[1].split("```", 1)[0].strip()
                data = json.loads(raw_data)

            logger.info(f"Moderation complete. Final P_YES: {data.get('p_yes')}")
            return ProbabilisticAnswer.model_validate(data)
        except Exception as e:
            logger.error(
                f"Failed to parse moderator response: {e}. Raw data: {moderator_resp.output}"
            )
            raise

    def predict(
        self,
        market_question: str,
        closing_date: str = "Unknown",
    ) -> Prediction:
        import asyncio
        import nest_asyncio
        nest_asyncio.apply()
        
        try:
            # 1. Research
            logger.info("PHASE 1: THE LIBRARIAN (Researching facts...)")
            research = self.research_agent.research(market_question)
            
            # 2. Debate
            logger.info("PHASE 2: THE EXPERTS (Starting adversarial debate...)")
            loop = asyncio.get_event_loop()
            answer = loop.run_until_complete(
                self.debate(
                    market_question,
                    research.report,
                    closing_date=closing_date,
                )
            )
            
            return Prediction(outcome_prediction=CategoricalProbabilisticAnswer.from_probabilistic_answer(answer))
        except Exception as e:
            logger.exception(f"Error in Luxembourg1Agent predict: {e}")
            return Prediction()
