import random


def exponential_backoff(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    delay = min(base * (2**attempt), cap)
    jitter = random.uniform(0.0, delay * 0.1)
    return delay + jitter
