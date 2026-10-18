"""Deterministic duration gate using capture time, independent of model training."""

import math


class SustainedFlag:
    def __init__(self, duration_s, max_gap_s):
        self.duration_s, self.max_gap_s = duration_s, max_gap_s
        self.reset()

    def reset(self):
        self.started = self.previous = None
        self.elapsed_s = 0.0

    def update(self, flagged, captured_s):
        if not math.isfinite(captured_s) or not flagged:
            self.reset()
            return False
        if (
            self.previous is None
            or captured_s <= self.previous
            or captured_s - self.previous > self.max_gap_s
        ):
            self.started = captured_s
        self.previous = captured_s
        self.elapsed_s = captured_s - self.started
        # User asked for longer than five seconds, so exactly five is not enough.
        return self.duration_s == 0 or self.elapsed_s > self.duration_s
