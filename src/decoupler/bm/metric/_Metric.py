from collections.abc import Callable


class Metric:
    def __init__(self, func: Callable, scores: list[str]):
        self.func = func
        self.scores = scores

    def __call__(self, **kwargs) -> tuple[float, float]:
        return self.func(**kwargs)
