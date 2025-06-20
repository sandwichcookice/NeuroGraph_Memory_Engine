import torch
from memory_engine import TextEncoder, ShortTermMemory, LongTermMemory, ReadNet, ActionDecoder, DecisionInterface, Consolidator, AttentionScheduler

def test_pipeline():
    enc = TextEncoder()
    stm = ShortTermMemory(embedding_dim=128)
    ltm = LongTermMemory(embedding_dim=128)
    readnet = ReadNet(state_dim=128)
    decoder = ActionDecoder(state_dim=128, hidden_dim=64, num_actions=3)
    consolidator = Consolidator()
    decider = DecisionInterface()

    state1 = enc('start')
    state2 = enc('move forward')
    s1 = stm.add_state(state1)
    s2 = stm.add_state(state2)
    stm.add_transition(s1, s2, 'forward')

    fused = readnet(state1, stm, ltm)
    logits = decoder(fused)
    assert logits.shape[-1] == 3

    consolidator.run(stm, ltm)
    action = decider.next_action(s1, s2, stm, ltm)
    assert action == 'forward'


def test_encoder_cache():
    enc = TextEncoder()
    emb1 = enc('hello world')
    size1 = len(enc.cache)
    emb2 = enc('hello world')
    size2 = len(enc.cache)
    assert size1 == 1
    assert size2 == 1
    assert torch.allclose(emb1, emb2)


def test_ltm_attention():
    state_dim = 2
    stm = ShortTermMemory(state_dim)
    ltm = LongTermMemory(state_dim)
    readnet = ReadNet(state_dim)

    query = torch.tensor([1.0, 0.0])
    n1 = ltm.add_state(torch.tensor([1.0, 0.0]))
    n2 = ltm.add_state(torch.tensor([0.0, 1.0]))
    root = ltm.add_state(query)
    ltm.add_transition(root, n1, 'a', 1.0)
    ltm.add_transition(root, n2, 'b', 1.0)

    msg = readnet.ltm_message(query, ltm)
    assert msg[0] > msg[1]


def test_plan_path():
    dim = 2
    stm = ShortTermMemory(dim)
    ltm = LongTermMemory(dim)
    decider = DecisionInterface()

    s0 = stm.add_state(torch.zeros(dim))
    s1 = stm.add_state(torch.ones(dim))
    s2 = stm.add_state(torch.tensor([2.0, 0.0]))
    stm.add_transition(s0, s1, 'go1')
    stm.add_transition(s1, s2, 'go2')

    path = decider.plan_path(s0, s2, stm, ltm)
    assert path == [(s1, 'go1'), (s2, 'go2')]

def test_attention_scheduler():
    dim = 4
    scheduler = AttentionScheduler(state_dim=dim)
    state = torch.ones(dim)
    units = torch.stack([torch.ones(dim), torch.zeros(dim)])
    output, idx = scheduler(state, units)
    assert idx == 0
