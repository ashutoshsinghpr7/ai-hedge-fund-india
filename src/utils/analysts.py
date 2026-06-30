"""Analyst configuration — single source of truth for all agents."""

from src.agents.warren_buffett import warren_buffett_agent
from src.agents.rakesh_jhunjhunwala import rakesh_jhunjhunwala_agent
from src.agents.radhakishan_damani import radhakishan_damani_agent
from src.agents.fii_dii_flow import fii_dii_flow_agent
from src.agents.rbi_policy import rbi_policy_agent
from src.agents.ben_graham import ben_graham_agent
from src.agents.bill_ackman import bill_ackman_agent
from src.agents.cathie_wood import cathie_wood_agent
from src.agents.charlie_munger import charlie_munger_agent
from src.agents.michael_burry import michael_burry_agent
from src.agents.peter_lynch import peter_lynch_agent
from src.agents.phil_fisher import phil_fisher_agent
from src.agents.stanley_druckenmiller import stanley_druckenmiller_agent
from src.agents.aswath_damodaran import aswath_damodaran_agent
from src.agents.mohnish_pabrai import mohnish_pabrai_agent
from src.agents.nassim_taleb import nassim_taleb_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.fundamentals import fundamentals_analyst_agent
from src.agents.sentiment import sentiment_analyst_agent
from src.agents.valuation import valuation_analyst_agent
from src.agents.growth_agent import growth_analyst_agent
from src.agents.news_sentiment import news_sentiment_agent

ANALYST_CONFIG = {
    "warren_buffett": {
        "display_name": "Warren Buffett",
        "description": "The Oracle of Omaha",
        "agent_func": warren_buffett_agent,
        "order": 0,
    },
    "rakesh_jhunjhunwala": {
        "display_name": "Rakesh Jhunjhunwala",
        "description": "The Big Bull of India",
        "agent_func": rakesh_jhunjhunwala_agent,
        "order": 1,
    },
    "radhakishan_damani": {
        "display_name": "Radhakishan Damani",
        "description": "The Retail Superinvestor",
        "agent_func": radhakishan_damani_agent,
        "order": 2,
    },
    "ben_graham": {
        "display_name": "Ben Graham",
        "description": "The Father of Value Investing",
        "agent_func": ben_graham_agent,
        "order": 3,
    },
    "bill_ackman": {
        "display_name": "Bill Ackman",
        "description": "Activist Value Investor",
        "agent_func": bill_ackman_agent,
        "order": 4,
    },
    "cathie_wood": {
        "display_name": "Cathie Wood",
        "description": "Disruptive Innovation Seeker",
        "agent_func": cathie_wood_agent,
        "order": 5,
    },
    "charlie_munger": {
        "display_name": "Charlie Munger",
        "description": "Mental Models & Compounders",
        "agent_func": charlie_munger_agent,
        "order": 6,
    },
    "michael_burry": {
        "display_name": "Michael Burry",
        "description": "The Big Short Contrarian",
        "agent_func": michael_burry_agent,
        "order": 7,
    },
    "peter_lynch": {
        "display_name": "Peter Lynch",
        "description": "Growth at a Reasonable Price",
        "agent_func": peter_lynch_agent,
        "order": 8,
    },
    "phil_fisher": {
        "display_name": "Phil Fisher",
        "description": "Quality Growth Scuttlebutt",
        "agent_func": phil_fisher_agent,
        "order": 9,
    },
    "stanley_druckenmiller": {
        "display_name": "Stanley Druckenmiller",
        "description": "Macro & Momentum Master",
        "agent_func": stanley_druckenmiller_agent,
        "order": 10,
    },
    "aswath_damodaran": {
        "display_name": "Aswath Damodaran",
        "description": "The Dean of Valuation",
        "agent_func": aswath_damodaran_agent,
        "order": 11,
    },
    "mohnish_pabrai": {
        "display_name": "Mohnish Pabrai",
        "description": "Dhando Investing Cloner",
        "agent_func": mohnish_pabrai_agent,
        "order": 12,
    },
    "nassim_taleb": {
        "display_name": "Nassim Taleb",
        "description": "Black Swan & Antifragility",
        "agent_func": nassim_taleb_agent,
        "order": 13,
    },
    "fii_dii_flow": {
        "display_name": "FII/DII Flow Analyst",
        "description": "Institutional Money Flow Tracker",
        "agent_func": fii_dii_flow_agent,
        "order": 14,
    },
    "rbi_policy": {
        "display_name": "RBI Policy Analyst",
        "description": "Monetary Policy Impact Specialist",
        "agent_func": rbi_policy_agent,
        "order": 15,
    },
    "technical_analyst": {
        "display_name": "Technical Analyst",
        "description": "Chart Pattern Specialist",
        "agent_func": technical_analyst_agent,
        "order": 16,
    },
    "fundamentals_analyst": {
        "display_name": "Fundamentals Analyst",
        "description": "Financial Statement Specialist",
        "agent_func": fundamentals_analyst_agent,
        "order": 17,
    },
    "sentiment_analyst": {
        "display_name": "Sentiment Analyst",
        "description": "Market Sentiment Specialist",
        "agent_func": sentiment_analyst_agent,
        "order": 18,
    },
    "valuation_analyst": {
        "display_name": "Valuation Analyst",
        "description": "Company Valuation Specialist",
        "agent_func": valuation_analyst_agent,
        "order": 19,
    },
    "growth_analyst": {
        "display_name": "Growth Analyst",
        "description": "Growth Specialist",
        "agent_func": growth_analyst_agent,
        "order": 20,
    },
    "news_sentiment": {
        "display_name": "News Sentiment Analyst",
        "description": "News Sentiment Specialist",
        "agent_func": news_sentiment_agent,
        "order": 21,
    },
}

ANALYST_ORDER = [(config["display_name"], key) for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])]


def get_analyst_nodes():
    return {key: (f"{key}_agent", config["agent_func"]) for key, config in ANALYST_CONFIG.items()}
