import json
import random
import sys
import os
import pickle
import torch
from memory_engine import (
    TextEncoder,
    ShortTermMemory,
    GNNLongTermMemory,
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
        if not isinstance(ltm, GNNLongTermMemory):
            ltm = GNNLongTermMemory(embed_dim)
else:
    ltm = GNNLongTermMemory(embed_dim)
readnet  = ReadNet(state_dim=128)
decider  = DecisionInterface(readnet)
decoder = ActionDecoder(state_dim=128, hidden_dim=64, num_actions=1)
consolidator = Consolidator()
cerebellum = Cerebellum()

# ---------------------------------------------------------------------------
# 預載通關攻略到 STM，初始化連結強度與獎勵皆為最高。
def preload_walkthrough() -> None:
    """建立從起點到取得鑽石鎬的基本流程，動作與觀察皆視為節點。"""

    pairs = [
        ('探索', '發現木材'),
        ('破壞方塊,手', '取得木材'),
        ('合成,木材', '取得木板'),
        ('合成,木棒', '取得木棒'),
        ('合成,工作台', '取得工作台'),
        ('合成,木鎬', '取得木鎬'),
        ('破壞方塊,木鎬', '取得石頭'),
        ('合成,石鎬', '取得石鎬'),
        ('破壞方塊,石鎬', '取得鐵礦'),
        ('合成,熔爐', '取得熔爐'),
        ('合成,鐵錠', '取得鐵錠'),
        ('合成,鐵鎬', '取得鐵鎬'),
        ('破壞方塊,鐵鎬', '取得鑽石'),
        ('合成,鑽石鎬', '取得鑽石鎬'),
    ]

    prev = stm.add_state(torch.randn(embed_dim), context_id=0, text='開始')
    for act, obs in pairs:
        # 動作節點
        act_node = stm.add_state(torch.randn(embed_dim), context_id=0, text=act)
        stm.add_transition(prev, act_node, act, reward=1.0)
        e1 = stm.graph[prev][act_node]
        e1['L'] = 1.0
        e1['R'] = 1.0
        e1['C'] = 1.0
        e1['weight'] = 3.0

        # 觀察節點
        obs_node = stm.add_state(torch.randn(embed_dim), context_id=0, text=obs)
        stm.add_transition(act_node, obs_node, f'觀察:{obs}', reward=1.0)
        e2 = stm.graph[act_node][obs_node]
        e2['L'] = 1.0
        e2['R'] = 1.0
        e2['C'] = 1.0
        e2['weight'] = 3.0
        prev = obs_node

    stm.graph.nodes[prev]['goal'] = '鑽石鎬'

preload_walkthrough()

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
visual_cycle = 10  # 每隔此步數輸出一次 STM 圖像
epsilon = 0.2

# 將可微邊權重同步回圖結構，供路徑規劃使用
def sync_ltm_weights():
    for key, param in ltm.edge_params.items():
        try:
            src, dst = map(int, key.split("->"))
            if ltm.graph.has_edge(src, dst):
                ltm.graph[src][dst]["weight"] = float(param.item())
        except Exception as e:  # pragma: no cover - 防止解析失敗
            print(f"sync error: {e}", file=sys.stderr)

# 根據記憶圖與小腦參數選擇下一步行動
def choose_action(state_repr: int):
    """
    優先使用記憶圖規劃路徑；若無路徑，再 fallback 到 ε-greedy + Cerebellum。
    """
    sync_ltm_weights()
    goal_node = ltm.find_goal("鑽石鎬")
    if goal_node:
        try:
            path = decider.plan_path(state_repr, goal_node)
        except Exception as e:  # pragma: no cover - 防止路徑規劃失敗
            print(f"plan error: {e}", file=sys.stderr)
            path = None
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

        raw = json.dumps(inv)
        emb = encoder(raw)
        node = stm.add_state(emb, context_id=0, text=raw)
        reward = 1.0 if inv != inventory_snapshot else 0.0
        if current_state is not None and current_action is not None:
            stm.add_transition(current_state, node, current_action, reward)
        current_state = node

        cerebellum.act(current_action, 1.0, reward)
        inventory_snapshot = inv

        step_counter += 1
        # 週期性輸出 STM 圖像，以供驗證
        if step_counter % visual_cycle == 0:
            try:
                out_path = f"stm_step_{step_counter}.png"
                stm.visualize(out_path)
            except Exception as e:
                print(f"visualize error: {e}", file=sys.stderr)

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
            try:
                with open(SNAPSHOT_PATH, "wb") as f:
                    pickle.dump(ltm, f)
            except Exception as e:
                print(f"save error: {e}", file=sys.stderr)
