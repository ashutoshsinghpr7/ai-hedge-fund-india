from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import json

from src.graph.state import AgentState, show_agent_reasoning
from src.tools import get_company_news
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress
from typing_extensions import Literal


class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: int = Field(description="Confidence 0-100")


def news_sentiment_agent(state: AgentState, agent_id: str = "news_sentiment_agent"):
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(ticker=ticker, end_date=end_date, limit=100, api_key=api_key)
        company_news = company_news or []
        news_signals = []
        sentiment_confidences = {}
        sentiments_classified_by_llm = 0

        if company_news:
            recent_articles = company_news[:10]
            articles_without_sentiment = [news for news in recent_articles if news.sentiment is None]
            if articles_without_sentiment:
                articles_to_analyze = articles_without_sentiment[:5]
                progress.update_status(agent_id, ticker, f"Analyzing sentiment for {len(articles_to_analyze)} articles")
                for idx, news in enumerate(articles_to_analyze):
                    prompt = (f"Please analyze the sentiment of the following news headline "
                              f"with the following context: The stock is {ticker}. "
                              f"Determine if sentiment is 'positive', 'negative', or 'neutral' for the stock {ticker} only. "
                              f"Also provide a confidence score for your prediction from 0 to 100. "
                              f"Respond in JSON format.\n\nHeadline: {news.title}")
                    response = call_llm(prompt, Sentiment, agent_name=agent_id, state=state)
                    if response:
                        news.sentiment = response.sentiment.lower()
                        sentiment_confidences[id(news)] = response.confidence
                    else:
                        news.sentiment = "neutral"
                        sentiment_confidences[id(news)] = 0
                    sentiments_classified_by_llm += 1

            sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
            news_signals = np.where(sentiment == "negative", "bearish",
                                    np.where(sentiment == "positive", "bullish", "neutral")).tolist()

        progress.update_status(agent_id, ticker, "Aggregating signals")
        bullish_signals = news_signals.count("bullish")
        bearish_signals = news_signals.count("bearish")
        neutral_signals = news_signals.count("neutral")

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        total_signals = len(news_signals)
        confidence = _calculate_confidence_score(sentiment_confidences, company_news, overall_signal,
                                                  bullish_signals, bearish_signals, total_signals)

        reasoning = {"news_sentiment": {"signal": overall_signal, "confidence": confidence,
                                         "metrics": {"total_articles": total_signals, "bullish_articles": bullish_signals,
                                                     "bearish_articles": bearish_signals, "neutral_articles": neutral_signals,
                                                     "articles_classified_by_llm": sentiments_classified_by_llm}}}

        sentiment_analysis[ticker] = {"signal": overall_signal, "confidence": confidence, "reasoning": reasoning}
        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    message = HumanMessage(content=json.dumps(sentiment_analysis), name=agent_id)
    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(sentiment_analysis, "News Sentiment Analysis Agent")
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"][agent_id] = sentiment_analysis
    progress.update_status(agent_id, None, "Done")
    return {"messages": [message], "data": state["data"]}


def _calculate_confidence_score(sentiment_confidences, company_news, overall_signal,
                                bullish_signals, bearish_signals, total_signals):
    if total_signals == 0:
        return 0.0
    if sentiment_confidences:
        matching_articles = [news for news in company_news if news.sentiment and (
            (overall_signal == "bullish" and news.sentiment == "positive") or
            (overall_signal == "bearish" and news.sentiment == "negative") or
            (overall_signal == "neutral" and news.sentiment == "neutral"))]
        llm_confidences = [sentiment_confidences[id(news)] for news in matching_articles if id(news) in sentiment_confidences]
        if llm_confidences:
            avg_llm_confidence = sum(llm_confidences) / len(llm_confidences)
            signal_proportion = (max(bullish_signals, bearish_signals) / total_signals) * 100
            return round(0.7 * avg_llm_confidence + 0.3 * signal_proportion, 2)
    return round((max(bullish_signals, bearish_signals) / total_signals) * 100, 2)
