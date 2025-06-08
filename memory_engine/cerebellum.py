class Cerebellum:
    """小腦模組，負責將動作意圖轉為具體控制參數，並透過回饋持續修正。"""

    def __init__(self, lr: float = 0.1, momentum: float = 0.5, default_strength: float = 1.0):
        # 學習率
        self.lr = lr
        # 動量係數，讓更新更穩定
        self.momentum = momentum
        self.default_strength = default_strength
        # 動作記憶：紀錄每個動作的執行強度與速度
        # {action: {"strength": float, "velocity": float}}
        self.memory: dict[str, dict[str, float]] = {}

    # ----------------------- Motor Generator -----------------------
    def motor_generator(self, action: str) -> float:
        """根據記憶產生初始控制參數。"""
        entry = self.memory.get(action)
        if entry is None:
            return self.default_strength
        return entry["strength"]

    # ----------------------- Error Comparator ----------------------
    def error_comparator(self, expected: float, actual: float) -> float:
        """計算期望與實際執行結果的差異。"""
        return expected - actual

    # ----------------------- Motor Corrector -----------------------
    def motor_corrector(self, action: str, error: float) -> float:
        """根據誤差更新動作記憶並回傳新的執行參數。"""
        entry = self.memory.setdefault(action, {"strength": self.default_strength, "velocity": 0.0})
        entry["velocity"] = self.momentum * entry["velocity"] + self.lr * error
        entry["strength"] += entry["velocity"]
        return entry["strength"]

    # ----------------------------- API -----------------------------
    def act(self, action: str, expected: float, actual: float | None = None) -> float:
        """產生動作參數，若有回饋則進行修正後回傳。"""
        param = self.motor_generator(action)
        if actual is not None:
            err = self.error_comparator(expected, actual)
            param = self.motor_corrector(action, err)
        return param

    def save(self) -> dict:
        """回傳可序列化的記憶結構。"""
        return self.memory

    def load(self, data: dict):
        """載入既有的小腦記憶。"""
        self.memory = {k: {"strength": float(v.get("strength", self.default_strength)), "velocity": float(v.get("velocity", 0.0))} for k, v in data.items()}
