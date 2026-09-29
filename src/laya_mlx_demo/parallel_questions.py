import laya_mlx as laya
import time


def main() -> None:
    agent = laya.load("aac6fef/laya-mlx")

    document = (
        "GDPR Article 33 requires personal data breaches to be reported to the supervisory "
        "authority without undue delay and, where feasible, not later than 72 hours after becoming aware. "
        "Maximum administrative fines can reach up to EUR 20 million or 4% of worldwide turnover."
    )

    questions = {
        "breach_72h": {
            "type": "noul",
            "instructions": "Must personal data breaches be reported within 72 hours?",
        },
        "max_fine": {
            "type": "choice",
            "instructions": "What is the maximum administrative fine under GDPR?",
            "criteria": {
                "TwentyM_or_4pct": "Up to EUR 20 million or 4% worldwide turnover",
                "TenM_or_2pct": "Up to EUR 10 million or 2% worldwide turnover",
                "FixedCap": "A fixed cap",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe are the financial penalties?",
            "criteria": [
                "None: No fines",
                "Minor: Small fixed fines",
                "Severe: Heavy turnover-based penalties",
            ],
        },
    }

    start = time.perf_counter()
    response = agent.predict(
        {"text": document},
        questions=questions,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Executed {len(questions)} parallel questions in {elapsed_ms:.1f} ms:")
    for key, answer in response["answers"].items():
        print(f" - {key}: {answer}")


if __name__ == "__main__":
    main()
