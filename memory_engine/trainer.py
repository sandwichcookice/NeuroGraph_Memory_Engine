import json
import random
import sys
from memory_engine import (
    TextEncoder,
    ShortTermMemory,
    LongTermMemory,
    ReadNet,
    ActionDecoder,
    DecisionInterface,
    Consolidator,
    Cerebellum,
)

# 初始化各模組
encoder = TextEncoder()
stm = ShortTermMemory(embedding_dim=128)
ltm = LongTermMemory(embedding_dim=128)
readnet = ReadNet(state_dim=128)
decoder = ActionDecoder(state_dim=128, hidden_dim=64, num_actions=1)
consolidator = Consolidator()
decider = DecisionInterface()
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

# 根據小腦參數選擇下一步行動
def choose_action():
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    weights = [cerebellum.motor_generator(a) for a in ACTIONS]
    total = sum(max(w, 0.0) for w in weights)
    if total == 0:
        return random.choice(ACTIONS)
    r = random.random() * total
    s = 0.0
    for a, w in zip(ACTIONS, weights):
        s += max(w, 0.0)
        if r <= s:
            return a
    return random.choice(ACTIONS)

for line in sys.stdin:
    data = line.strip()
    if not data:
        continue
    if data == 'start':
        act = choose_action()
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
        print('exit')
        sys.stdout.flush()
        break

    act = choose_action()
    current_action = act
    print(act)
    sys.stdout.flush()
