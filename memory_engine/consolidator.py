from .stm import ShortTermMemory
from .ltm import LongTermMemory

class Consolidator:
    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def run(self, stm: ShortTermMemory, ltm: LongTermMemory):
        ltm.consolidate(stm, self.beta)
        stm.decay_and_prune()
        ltm.decay_and_prune()
