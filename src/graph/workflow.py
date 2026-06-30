"""LangGraph workflow builder for the multi-agent hedge fund."""

from langgraph.graph import END, StateGraph

from src.graph.state import AgentState
from src.agents.risk_manager import risk_management_agent
from src.agents.portfolio_manager import portfolio_management_agent
from src.utils.analysts import get_analyst_nodes


def start(state: AgentState):
    return state


def create_workflow(selected_analysts=None):
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    analyst_nodes = get_analyst_nodes()
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())

    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start_node")
    return workflow
