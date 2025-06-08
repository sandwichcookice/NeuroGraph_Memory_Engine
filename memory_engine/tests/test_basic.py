import torch
from memory_engine import TextEncoder, ShortTermMemory, LongTermMemory, ReadNet, ActionDecoder, DecisionInterface, Consolidator

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
