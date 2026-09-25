import time

import laya_mlx as laya


def main() -> None:
    agent = laya.load("aac6fef/laya-mlx")
    start_time = time.perf_counter()
    result = agent.predict(
        "I was billed twice. Please refund the duplicate today.",
        {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this request?",
                "criteria": ["billing", "technical", "sales"],
            },
            "refund": {
                "type": "noul",
                "instructions": "Does the customer ask for money back?",
            },
        },
    )
    duration_ms = (time.perf_counter() - start_time) * 1000
    print(result["answers"])
    print(f"Time spent: {duration_ms:.2f} ms")


if __name__ == "__main__":
    main()
