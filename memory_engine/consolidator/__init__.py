import logging
from ..stm import ShortTermMemory
from ..ltm import LongTermMemory

logger = logging.getLogger(__name__)

class Consolidator:
    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def run(self, stm: ShortTermMemory, ltm: LongTermMemory):
        try:
            ltm.consolidate(stm, self.beta)
            stm.decay_and_prune()
            ltm.decay_and_prune()
        except Exception as e:
            logger.exception("鞏固流程失敗: %s", e)
            raise
