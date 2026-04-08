import os
import logging
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent

# Silence Langfuse warnings if keys aren't set
logging.getLogger("langfuse").setLevel(logging.ERROR)
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    FullBinaryKellyBettingStrategy,
)
from prediction_market_agent_tooling.deploy.betting_strategy import (
    MaxAccuracyWithKellyScaledBetsStrategy,
)
from prediction_market_agent_tooling.gtypes import USD
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket, SortBy
from prediction_market_agent_tooling.markets.data_models import (
    ProbabilisticAnswer,
)
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.openai_utils import get_openai_provider
from prediction_market_agent_tooling.tools.relevant_news_analysis.relevant_news_analysis import (
    get_certified_relevant_news_since_cached,
)
from prediction_market_agent_tooling.tools.relevant_news_analysis.relevant_news_cache import (
    RelevantNewsResponseCache,
)
from prediction_market_agent_tooling.tools.utils import utcnow
from prediction_prophet.benchmark.agents import (
    PredictionProphetAgent,
)
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings

from prediction_market_agent.agents.luxembourg1_agent.luxembourg1_agent import Luxembourg1Agent
from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount
from prediction_market_agent.utils import (
    APIKeys,
)

class DeployableLuxembourg1Agent(DeployableTraderAgent):
    """
    Luxembourg1 Architecture:
    1. Librarian (Prophet Research): Uses Prophet to crawl the web and build a report.
    2. Expert Debate (Luxembourg1): Bull and Bear agents debate the report.
    3. Final Moderation: A moderator weights the debate for a final answer.
    4. Staleness Check: Only trades if new info emerged since the last trade.
    """
    agent: Luxembourg1Agent
    
    # Default to closing soonest so we test markets that resolve first.
    # You can override counts via env vars.
    get_markets_sort_by = SortBy.CLOSING_SOONEST

    # Defaults can be overridden via env vars.
    n_markets_to_fetch = int(os.environ.get("LUXEMBOURG1_N_MARKETS_TO_FETCH", "10"))
    bet_on_n_markets_per_run = int(
        os.environ.get("LUXEMBOURG1_BET_ON_N_MARKETS_PER_RUN", "10")
    )
    
    # --- Profitability Filters (Configurable via env vars) ---
    CONFIDENCE_THRESHOLD = float(os.environ.get("LUXEMBOURG1_CONFIDENCE_THRESHOLD", "0.7"))
    INDECISION_BUFFER = float(os.environ.get("LUXEMBOURG1_INDECISION_BUFFER", "0.1")) # 0.1 = skip 0.4 to 0.6

    def load(self) -> None:
        super().load()
        # Defaults: efficient + strong (can be overridden via env vars).
        model = os.environ.get("LUXEMBOURG1_MODEL", "gpt-5.4-mini")
        reasoning_model = os.environ.get("LUXEMBOURG1_REASONING_MODEL", "gpt-5.4")
        api_keys = APIKeys()
        
        # Staleness Detection Cache
        if not api_keys.SQLALCHEMY_DB_URL:
            logger.info("SQLALCHEMY_DB_URL not found. Using local sqlite for cache.")
            from pydantic import SecretStr
            api_keys.SQLALCHEMY_DB_URL = SecretStr("sqlite:///luxembourg1_cache.db")

        self.relevant_news_response_cache = RelevantNewsResponseCache(api_keys=api_keys)

        # ---------------------------------------------------------------------
        # THE LIBRARIAN (Prophet Research Agent)
        # ---------------------------------------------------------------------
        # This agent uses tools (web search, scraping) to gather facts.
        openai_model = OpenAIChatModel(
            model,
            provider=get_openai_provider(api_key=api_keys.openai_api_key),
        )
        # Use a dedicated reasoning model (o3-mini) for the final moderation step,
        # as suggested in the Prophet++ blueprint.
        reasoning_openai_model = OpenAIChatModel(
            reasoning_model,
            provider=get_openai_provider(api_key=api_keys.openai_api_key),
        )

        prophet_librarian = PredictionProphetAgent(
            research_agent=Agent(
                openai_model,
                model_settings=ModelSettings(temperature=0.7),
            ),
            # We don't use Prophet's final answer, so this is just filler.
            prediction_agent=Agent(
                openai_model,
                model_settings=ModelSettings(temperature=0.0),
            ),
            include_reasoning=True,
            logger=logger,
        )

        # ---------------------------------------------------------------------
        # THE EXPERTS (Luxembourg1 Debate Logic)
        # ---------------------------------------------------------------------
        # This takes the Librarian's facts and runs the adversarial reasoning.
        # We pass the same 'openai_model' so it has the API key.
        self.agent = Luxembourg1Agent(
            research_agent=prophet_librarian,
            debate_llm=openai_model,
            moderator_llm=reasoning_openai_model,
        )

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
    return MaxAccuracyWithKellyScaledBetsStrategy(
        max_position_amount=get_maximum_possible_bet_amount(
            min_=USD(0.001),
            max_=USD(0.01),
            trading_balance=market.get_trade_balance(APIKeys()),
        ),
    )

    def verify_market(self, market_type: MarketType, market: AgentMarket) -> bool:
        if not super().verify_market(market_type, market):
            return False

        # Staleness Detection:
        # Only trade if we have fresh news since our last trade, 
        # or if we have never traded on this market before.
        user_id = market.get_user_id(api_keys=APIKeys())
        last_trade_datetime = market.get_most_recent_trade_datetime(user_id=user_id)
        
        if last_trade_datetime is None:
            logger.info(f"No previous trades for market '{market.question}'. Proceeding.")
            return True

        # Check for relevant news since last trade
        news = get_certified_relevant_news_since_cached(
            question=market.question,
            days_ago=max((utcnow() - last_trade_datetime).days, 1),
            cache=self.relevant_news_response_cache,
        )
        
        if news is not None:
            logger.info(f"Fresh news detected for '{market.question}'. Re-evaluating.")
            return True
        
        logger.info(f"No fresh news for '{market.question}' since last trade. Skipping.")
        return False

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        closing_date = (
            market.close_time.strftime("%Y-%m-%d")
            if market.close_time is not None
            else "Unknown"
        )

        prediction = self.agent.predict(
            market.question,
            closing_date=closing_date,
        )
        if prediction is None or prediction.outcome_prediction is None:
            return None
            
        # 1. Confidence Filter
        p_yes = prediction.outcome_prediction.get_yes_probability()
        if p_yes is None:
             logger.warning(f"Could not find YES probability for '{market.question[:50]}...'. Skipping.")
             return None

        confidence = prediction.outcome_prediction.confidence

        if confidence < self.CONFIDENCE_THRESHOLD:
            logger.info(
                f"Skipping '{market.question[:50]}...': Confidence too low ({confidence:.2f} < {self.CONFIDENCE_THRESHOLD})"
            )
            return None

        # 2. Indecision Filter (Probability too close to 0.5)
        p_yes_float = float(p_yes)
        if (0.5 - self.INDECISION_BUFFER) < p_yes_float < (0.5 + self.INDECISION_BUFFER):
            logger.info(
                f"Skipping '{market.question[:50]}...': Probabilities too close to coin toss (P_YES={p_yes_float:.2f})"
            )
            return None

        # 3. Force p_yes to the agent's committed side so Kelly never bets opposite.
        # A tiny nudge (+/- 0.001) ensures p_yes is strictly on the correct side of
        # the market price, so Kelly sizes the bet in the direction the agent believes
        # without ever flipping direction due to a near-neutral spread.
        market_p_yes = float(market.p_yes)
        if p_yes_float >= 0.5:
            committed_p_yes = max(p_yes_float, market_p_yes + 0.001)
        else:
            committed_p_yes = min(p_yes_float, market_p_yes - 0.001)

        logger.info(
            f"Market P_YES={market_p_yes:.3f}, Agent P_YES={p_yes_float:.3f} → "
            f"Committed P_YES={committed_p_yes:.3f} → betting {'YES' if committed_p_yes >= 0.5 else 'NO'}"
        )

        answer = ProbabilisticAnswer(
            p_yes=committed_p_yes,
            confidence=confidence,
        )
        logger.info(
            f"Returning answer: p_yes={float(answer.p_yes):.3f}, confidence={answer.confidence:.2f} → betting {'YES' if float(answer.p_yes) >= 0.5 else 'NO'}"
        )
        return answer

