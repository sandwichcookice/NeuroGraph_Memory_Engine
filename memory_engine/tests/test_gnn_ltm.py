import torch
from memory_engine.ltm_gnn import GNNLongTermMemory
from memory_engine.stm import ShortTermMemory
from memory_engine.consolidator import Consolidator


def test_gnn_ltm_training():
    gnn = GNNLongTermMemory(embedding_dim=4, hidden_dim=8)
    s0 = gnn.add_state(torch.zeros(4))
    s1 = gnn.add_state(torch.ones(4))
    gnn.add_edge(s0, s1, 'go')
    samples = [(s0, 'go', s1, 1.0)]
    before = gnn.train_offline(samples, epochs=1)
    after = gnn.train_offline(samples, epochs=10)
    assert after <= before


def test_consolidator_with_gnn():
    stm = ShortTermMemory(embedding_dim=2)
    gnn = GNNLongTermMemory(embedding_dim=2, hidden_dim=4)
    cons = Consolidator(beta=0.1)
    n0 = stm.add_state(torch.zeros(2))
    n1 = stm.add_state(torch.ones(2))
    stm.add_transition(n0, n1, 'go', reward=1.0)
    cons.run(stm, gnn)
    assert gnn.graph.has_edge(n0, n1)
