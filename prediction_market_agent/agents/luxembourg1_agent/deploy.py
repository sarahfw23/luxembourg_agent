import os
import logging
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent

# Silence Langfuse warnings if keys aren't set
logging.getLogger("langfuse").setLevel(logging.ERROR)
if os.environ.get("DRY_RUN") == "1":
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    FullBinaryKellyBettingStrategy,
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
from pydantic_ai.models.openai import OpenAIModel
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
    
    # In dry run, we strictly only fetch and process 1 market to save credits.
    n_markets_to_fetch = 1 if os.environ.get("DRY_RUN") == "1" else 10
    bet_on_n_markets_per_run = 1 # only for dry run
    get_markets_sort_by = SortBy.NEWEST

    def load(self) -> None:
        super().load()
        model = "gpt-4o-2024-08-06"
        reasoning_model = "o3-mini-2025-01-31"
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
        openai_model = OpenAIModel(
            model,
            provider=get_openai_provider(api_key=api_keys.openai_api_key),
        )
        # Use a dedicated reasoning model (o3-mini) for the final moderation step,
        # as suggested in the Prophet++ blueprint.
        reasoning_openai_model = OpenAIModel(
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
        # Optimized for ~50 cent balance:
        return FullBinaryKellyBettingStrategy(
            max_position_amount=get_maximum_possible_bet_amount(
                min_=USD(0.01),
                max_=USD(0.50),
                trading_balance=market.get_trade_balance(APIKeys()),
            ),
            max_price_impact=0.05, 
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

    def check_min_required_balance_to_trade(self, market: AgentMarket) -> None:
        if os.environ.get("DRY_RUN") == "1":
            logger.info("DRY_RUN=1: Bypassing balance check for Luxembourg1.")
            return
        return super().check_min_required_balance_to_trade(market)


    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        prediction = self.agent.predict(market.question)
        if prediction is None or prediction.outcome_prediction is None:
            return None
        
        logger.info(
            f"Luxembourg1 Answering '{market.question}' with '{prediction.outcome_prediction}'."
        )
        return prediction.outcome_prediction.to_probabilistic_answer()
