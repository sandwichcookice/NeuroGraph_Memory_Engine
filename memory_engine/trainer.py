import json
import random
import sys
import os
import pickle
from memory_engine import (
    TextEncoder,
    ShortTermMemory,
    LongTermMemory,
    ActionDecoder,
    Consolidator,
    Cerebellum,
)
from memory_engine.decision import DecisionInterface, ReadNet

SNAPSHOT_PATH = "ltm_snapshot.pkl"

# 初始化各模組
encoder = TextEncoder()
stm = ShortTermMemory(embedding_dim=128)
embed_dim = 128
if os.path.exists(SNAPSHOT_PATH):
    with open(SNAPSHOT_PATH, "rb") as f:
        ltm = pickle.load(f)
else:
    ltm = LongTermMemory(embed_dim)
readnet  = ReadNet()
decider  = DecisionInterface(readnet, stm, ltm)
decoder = ActionDecoder(state_dim=128, hidden_dim=64, num_actions=1)
consolidator = Consolidator()
cerebellum = Cerebellum()

# 可選指令集合
ACTIONS = [
    '探索',
    '破壞方塊,手',
    '破壞方塊,木鎬',
    '破壞方塊,石鎬',
    '破壞方塊,鐵鎬',
    '合成,木材',
    '合成,木棒',
    '合成,工作台',
    '合成,木鎬',
    '合成,石鎬',
    '合成,熔爐',
    '合成,鐵錠',
    '合成,鐵鎬',
    '合成,鑽石鎬',
]

current_state = None
current_action = None
inventory_snapshot = {}
step_counter = 0
sleep_cycle = 20
epsilon = 0.2

# 根據記憶圖與小腦參數選擇下一步行動
def choose_action(state_repr: int):
    """
    優先使用記憶圖規劃路徑；若無路徑，再 fallback 到 ε-greedy + Cerebellum。
    """
    goal_node = ltm.find_goal("鑽石鎬")
    if goal_node:
        path = decider.plan_path(state_repr, goal_node)
        if path:
            return path[0][1]
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return cerebellum.best_action()


def main():
    global current_state, current_action, inventory_snapshot, step_counter
    for line in sys.stdin:
        data = line.strip()
        if not data:
            continue
        if data == 'start':
            act = choose_action(current_state if current_state is not None else 0)
            current_action = act
            print(act)
            sys.stdout.flush()
            continue
        payload = json.loads(data)
        inv = payload.get('inventory', {})
        response = payload.get('response', '')

        emb = encoder(json.dumps(inv))
        node = stm.add_state(emb)
        if current_state is not None and current_action is not None:
            stm.add_transition(current_state, node, current_action)
        current_state = node

        reward = 1.0 if inv != inventory_snapshot else 0.0
        cerebellum.act(current_action, 1.0, reward)
        inventory_snapshot = inv

        step_counter += 1
        if step_counter % sleep_cycle == 0:
            consolidator.run(stm, ltm)

        if '鑽石鎬' in inv:
            ltm.graph.nodes[node]["goal"] = "鑽石鎬"
            with open(SNAPSHOT_PATH, "wb") as f:
                pickle.dump(ltm, f)
            print('exit')
            sys.stdout.flush()
            break

        act = choose_action(current_state)
        current_action = act
        print(act)
        sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    finally:
        if ltm is not None:
            with open(SNAPSHOT_PATH, "wb") as f:
                pickle.dump(ltm, f)
