"""
A small Laya-MLX backed ticket router with application-owned policy.
"""

import json
import laya_mlx as laya
import time
from dataclasses import asdict, dataclass
from typing import Literal

Route = Literal["bypass_closed", "review", "no_action", "simulate_priority_queue"]


@dataclass(frozen=True)
class Ticket:
    status: str
    message: str
    duplicate_charge_confirmed: bool


@dataclass(frozen=True)
class DecisionSignals:
    billing_probability: float
    category: str
    category_confidence: float
    urgency_score: float
    urgency_confidence: float


@dataclass(frozen=True)
class PolicyThresholds:
    min_billing_probability: float = 0.70
    min_category_confidence: float = 0.80
    min_urgency_score: float = 1.50
    min_urgency_confidence: float = 0.75


def deterministic_route(ticket: Ticket) -> Route | None:
    """Handle rules that do not need a model."""
    if ticket.status == "closed":
        return "bypass_closed"
    if not ticket.duplicate_charge_confirmed:
        return "review"
    return None


def evaluate_ticket(ticket: Ticket, agent: laya.Agent | None = None) -> DecisionSignals:
    """Ask Laya-MLX three typed questions over one shared state."""
    if agent is None:
        agent = laya.load("aac6fef/laya-mlx")

    state = {
        "message": ticket.message,
        "duplicate_charge_confirmed": ticket.duplicate_charge_confirmed,
        "policy": (
            "Billing can investigate confirmed duplicate subscription charges."
        ),
    }

    start_time = time.perf_counter()
    response = agent.predict(
        state=state,
        questions={
            "is_billing_issue": {
                "type": "noul",
                "instructions": "Is this ticket primarily about a billing problem?",
            },
            "ticket_category": {
                "type": "choice",
                "instructions": "Which category best describes the ticket?",
                "criteria": {
                    "billing": "Charges, invoices, subscriptions, or refunds",
                    "technical": "Product defects, outages, or integrations",
                    "account": "Login, access, or account administration",
                    "other": "None of the listed categories fits",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How time-sensitive is the requested response?",
                "criteria": [
                    "No stated time pressure",
                    "Prompt response requested, but no active harm",
                    "Active financial or operational harm is described",
                ],
            },
        },
    )
    duration_ms = (time.perf_counter() - start_time) * 1000

    answers = response["answers"]
    category = answers["ticket_category"]
    urgency = answers["urgency"]
    billing = answers["is_billing_issue"]

    print(f"Time spent: {duration_ms:.2f} ms")
    return DecisionSignals(
        billing_probability=billing["noul"],
        category=category["choice"],
        category_confidence=category["confidence"],
        urgency_score=urgency["score"],
        urgency_confidence=urgency["confidence"],
    )


def route_decision(
        ticket: Ticket,
        signals: DecisionSignals,
        thresholds: PolicyThresholds = PolicyThresholds(),
) -> Route:
    """Convert model signals into an application-owned route."""
    deterministic = deterministic_route(ticket)
    if deterministic is not None:
        return deterministic

    if signals.category_confidence < thresholds.min_category_confidence:
        return "review"

    if (
            signals.category != "billing"
            or signals.billing_probability < thresholds.min_billing_probability
    ):
        return "no_action"

    if signals.urgency_confidence < thresholds.min_urgency_confidence:
        return "review"

    if signals.urgency_score >= thresholds.min_urgency_score:
        return "simulate_priority_queue"

    return "review"


def main() -> None:
    ticket = Ticket(
        status="open",
        message="A duplicate subscription charge appeared this month.",
        duplicate_charge_confirmed=True,
    )
    thresholds = PolicyThresholds()

    deterministic = deterministic_route(ticket)
    if deterministic is not None:
        result = {"route": deterministic, "source": "deterministic"}
    else:
        agent = laya.load("aac6fef/laya-mlx")
        signals = evaluate_ticket(ticket, agent=agent)
        result = {
            "route": route_decision(ticket, signals, thresholds),
            "source": "laya_then_policy",
            "signals": asdict(signals),
            "thresholds": asdict(thresholds),
        }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
