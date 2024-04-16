from functools import total_ordering
from typing import Self

from open_alex_interface import Work


@total_ordering
class ScoredWork:
    def __init__(self, work: Work, score: float):
        self.work = work
        self.score = score

    @property
    def score(self) -> float:
        return self._score

    @score.setter
    def score(self, value: float):
        if not isinstance(value, float) or value < 0.0 or value > 1.0:
            raise ValueError("Score must be a float.")
        self._score = value

    def __str__(self) -> str:
        return f"Score: {self.score} - {self.work}"

    def __eq__(self, other: Self):
        return self.score == other.score

    def __lt__(self, other: Self):
        return self.score < other.score
