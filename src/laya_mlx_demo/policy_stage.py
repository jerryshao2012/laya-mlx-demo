import laya_mlx as laya
import time

state = {
    "message": "A duplicate subscription charge appeared this month.",
    "duplicate_charge_confirmed": True,
    "policy": "Billing can investigate confirmed duplicate subscription charges.",
}


def main() -> None:
    agent = laya.load("aac6fef/laya-mlx")

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
    billing = answers["is_billing_issue"]
    category = answers["ticket_category"]
    urgency = answers["urgency"]

    print("billing_probability:", billing["noul"])
    print("category:", category["choice"])
    print("category_probabilities:", category["probabilities"])
    print("category_confidence:", category["confidence"])
    print("urgency_score:", urgency["score"])
    print("urgency_probabilities:", urgency["probabilities"])
    print("urgency_confidence:", urgency["confidence"])
    print(f"Time spent: {duration_ms:.2f} ms")


if __name__ == "__main__":
    main()
