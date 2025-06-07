import networkx as nx
from .stm import ShortTermMemory
from .ltm import LongTermMemory

class DecisionInterface:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def build_graph(self, stm: ShortTermMemory, ltm: LongTermMemory) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(stm.graph.nodes(data=True))
        g.add_nodes_from(ltm.graph.nodes(data=True))
        for u, v, data in stm.graph.edges(data=True):
            weight = self.alpha * data['weight']
            g.add_edge(u, v, action=data['action'], weight=weight)
        for u, v, data in ltm.graph.edges(data=True):
            weight = (1 - self.alpha) * data['weight']
            if g.has_edge(u, v):
                g[u][v]['weight'] += weight
            else:
                g.add_edge(u, v, action=data['action'], weight=weight)
        return g

    def next_action(self, current: int, goal: int, stm: ShortTermMemory, ltm: LongTermMemory):
        g = self.build_graph(stm, ltm)
        try:
            path = nx.shortest_path(g, source=current, target=goal, weight=lambda u,v,d: 1/d['weight'])
        except nx.NetworkXNoPath:
            return None
        if len(path) < 2:
            return None
        action = g[path[0]][path[1]]['action']
        return action
