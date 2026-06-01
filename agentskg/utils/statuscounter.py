class StatsCounter:
    def __init__(self):
        self.llm_calls = 0

    def increment(self):
        self.llm_calls += 1

    def reset(self):
        self.llm_calls = 0

global_stats = StatsCounter()
