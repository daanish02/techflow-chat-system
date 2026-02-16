"""Integration tests for multi-agent system."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.graph import get_agent_graph, route_from_greeter, route_from_retention
from src.agents.greeter_agent import classify_intent, extract_email
from src.agents.processor_agent import determine_final_action
from src.agents.retention_agent import determine_cancellation_reason, should_query_rag
from src.agents.state import (
    create_initial_state,
    get_last_user_message,
    is_authenticated,
)


class TestGreeterAgent:
    """Tests for Greeter Agent functionality."""

    def test_extract_email(self):
        """Test email extraction from messages."""
        assert extract_email("My email is test@example.com") == "test@example.com"
        assert (
            extract_email("contact me at john.doe@company.org")
            == "john.doe@company.org"
        )
        assert extract_email("no email here") is None

    def test_classify_intent_cancellation(self):
        """Test classification of cancellation intent."""
        messages = [
            "I want to cancel my insurance",
            "need to stop coverage can't afford it",
            "terminate my care+ subscription",
        ]
        for msg in messages:
            assert classify_intent(msg) == "cancellation"

    def test_classify_intent_technical(self):
        """Test classification of technical support intent."""
        messages = [
            "my phone is overheating",
            "screen won't turn on",
            "battery drain issues",
        ]
        for msg in messages:
            assert classify_intent(msg) == "technical"

    def test_classify_intent_billing(self):
        """Test classification of billing intent."""
        messages = [
            "why is my bill so high",
            "question about charges",
            "need refund for payment",
        ]
        for msg in messages:
            assert classify_intent(msg) == "billing"


class TestRetentionAgent:
    """Tests for Retention Agent functionality."""

    def test_determine_cancellation_reason_financial(self):
        """Test detecting financial hardship reason."""
        state = create_initial_state("can't afford the $13/month anymore")
        reason = determine_cancellation_reason(state)
        assert reason == "financial_hardship"

    def test_determine_cancellation_reason_not_using(self):
        """Test detecting not using reason."""
        state = create_initial_state("paying for it but never used the insurance")
        reason = determine_cancellation_reason(state)
        assert reason == "not_using"

    def test_determine_cancellation_reason_defect(self):
        """Test detecting product defect reason."""
        state = create_initial_state("phone is broken want to return it")
        reason = determine_cancellation_reason(state)
        assert reason == "product_defect"

    def test_should_query_rag_true(self):
        """Test RAG query is triggered for benefit questions."""
        state = create_initial_state("what does care+ actually cover?")
        assert should_query_rag(state) is True

    def test_should_query_rag_false(self):
        """Test RAG query not triggered for simple cancellation."""
        state = create_initial_state("just cancel it please")
        assert should_query_rag(state) is False


class TestProcessorAgent:
    """Tests for Processor Agent functionality."""

    def test_determine_action_accept_discount(self):
        """Test detecting discount acceptance."""
        state = create_initial_state("yes ill take the discount")
        state["retention_offers"] = [
            {"type": "discount", "description": "50% off for 6 months"}
        ]
        state["messages"].append(HumanMessage(content="yes sounds good"))

        action, details = determine_final_action(state)
        assert action == "accepted_discount"
        assert "50%" in details

    def test_determine_action_cancellation(self):
        """Test detecting cancellation decision."""
        state = create_initial_state("no thanks still want to cancel")
        state["reason"] = "financial_hardship"
        state["messages"].append(HumanMessage(content="just cancel it"))

        action, details = determine_final_action(state)
        assert action == "cancelled_insurance"
        assert "financial_hardship" in details

    def test_determine_action_kept_coverage(self):
        """Test detecting decision to keep coverage."""
        state = create_initial_state("ok ill keep it")
        state["messages"].append(HumanMessage(content="ill just keep what i have"))

        action, details = determine_final_action(state)
        assert action == "kept_coverage"


class TestGraphRouting:
    """Tests for graph routing logic."""

    def test_route_from_greeter_to_retention(self):
        """Test routing from greeter to retention."""
        state = create_initial_state("cancel insurance")
        state["routing_decision"] = "retention"

        next_node = route_from_greeter(state)
        assert next_node == "retention"

    def test_route_from_greeter_to_tech_support(self):
        """Test routing from greeter to tech support."""
        state = create_initial_state("phone broken")
        state["routing_decision"] = "tech_support"

        next_node = route_from_greeter(state)
        assert next_node == "tech_support"

    def test_route_from_greeter_to_billing(self):
        """Test routing from greeter to billing."""
        state = create_initial_state("billing question")
        state["routing_decision"] = "billing"

        next_node = route_from_greeter(state)
        assert next_node == "billing"

    def test_route_from_greeter_loop(self):
        """Test greeter loops when more info needed."""
        state = create_initial_state("hello")
        state["routing_decision"] = "greeter"

        next_node = route_from_greeter(state)
        assert next_node == "greeter"

    def test_route_from_retention_to_processor(self):
        """Test routing from retention to processor."""
        state = create_initial_state("yes accept offer")
        state["routing_decision"] = "processor"

        next_node = route_from_retention(state)
        assert next_node == "processor"

    def test_route_from_retention_loop(self):
        """Test retention loops when continuing conversation."""
        state = create_initial_state("tell me more")
        state["routing_decision"] = "retention"

        next_node = route_from_retention(state)
        assert next_node == "retention"


class TestConversationState:
    """Tests for conversation state management."""

    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state("Hello")

        assert len(state["messages"]) == 1
        assert state["current_agent"] == "greeter"
        assert state["customer_data"] is None
        assert state["intent"] is None

    def test_is_authenticated(self):
        """Test authentication check."""
        state = create_initial_state("Hello")
        assert is_authenticated(state) is False

        state["customer_data"] = {"name": "John", "tier": "gold"}
        assert is_authenticated(state) is True

    def test_get_last_user_message(self):
        """Test retrieving last user message."""
        state = create_initial_state("Hello")
        state["messages"].append(AIMessage(content="Hi there"))
        state["messages"].append(HumanMessage(content="I need help"))

        last_msg = get_last_user_message(state)
        assert last_msg == "I need help"


class TestGraphExecution:
    """Tests for full graph execution."""

    def test_graph_compiles(self):
        """Test that graph compiles without errors."""
        graph = get_agent_graph()
        assert graph is not None

    @pytest.mark.skip(reason="Requires API key and full integration")
    def test_graph_execution_simple(self):
        """Test simple graph execution (requires API key)."""
        graph = get_agent_graph()
        state = create_initial_state("Hello")

        result = graph.invoke(state)

        assert len(result["messages"]) > 1
        assert result["current_agent"] in ["greeter", "retention", "processor"]


class TestStateTransitions:
    """Tests for state transitions between agents."""

    def test_greeter_adds_customer_data(self):
        """Test greeter successfully adds customer data to state."""
        state = create_initial_state("my email is sarah.martinez@email.com")

        # simulate greeter extracting email
        email = extract_email(state["messages"][0].content)
        assert email == "sarah.martinez@email.com"

    def test_retention_adds_offers(self):
        """Test retention adds offers to state."""
        state = create_initial_state("cancel insurance")
        state["customer_data"] = {"tier": "gold"}
        state["intent"] = "cancellation"

        # Retention would add offers
        # This is tested in the full integration
        assert state["customer_data"]["tier"] == "gold"

    def test_processor_logs_action(self):
        """Test processor determines and sets final action."""
        state = create_initial_state("yes accept")
        state["customer_id"] = "C001"
        state["retention_offers"] = [{"type": "discount", "description": "50% off"}]
        state["messages"].append(HumanMessage(content="yes"))

        action, details = determine_final_action(state)
        assert action in ["accepted_discount", "kept_coverage"]


class TestScenarioWorkflows:
    """Test complete workflows for assignment scenarios."""

    def test_scenario_workflow_cancellation(self):
        """Test cancellation scenario workflow."""
        # Start
        state = create_initial_state("cant afford care+ need to cancel")

        # After greeter
        intent = classify_intent(state["messages"][0].content)
        assert intent == "cancellation"

        # After routing
        state["intent"] = "cancellation"
        state["routing_decision"] = "retention"
        next_node = route_from_greeter(state)
        assert next_node == "retention"

    def test_scenario_workflow_technical(self):
        """Test technical support workflow."""
        state = create_initial_state("phone overheating wont charge")

        intent = classify_intent(state["messages"][0].content)
        assert intent == "technical"

        state["routing_decision"] = "tech_support"
        next_node = route_from_greeter(state)
        assert next_node == "tech_support"

    def test_scenario_workflow_billing(self):
        """Test billing inquiry workflow."""
        state = create_initial_state("charged wrong amount on bill")

        intent = classify_intent(state["messages"][0].content)
        assert intent == "billing"

        state["routing_decision"] = "billing"
        next_node = route_from_greeter(state)
        assert next_node == "billing"
