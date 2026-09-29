import laya_mlx as laya
import time


def main() -> None:
    agent = laya.load("aac6fef/laya-mlx")

    start_time = time.perf_counter()
    response = agent.predict(
        "A duplicate subscription charge appeared this month.",
        {
            "is_billing_issue": {
                "type": "noul",
                "instructions": "Is this ticket primarily about a billing problem?",
            }
        },
    )
    duration_ms = (time.perf_counter() - start_time) * 1000

    print("Billing probability (noul):", response["answers"]["is_billing_issue"]["noul"])
    print(f"Time spent: {duration_ms:.2f} ms")


if __name__ == "__main__":
    main()
