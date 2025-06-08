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
        path = self.plan_path(current, goal, stm, ltm)
        if not path:
            return None
        return path[0][1]

    def plan_path(self, current: int, goal: int, stm: ShortTermMemory, ltm: LongTermMemory):
        """回傳從 current 到 goal 的行動路徑。

        Returns a list of tuples ``(node, action)`` 表示依序經過的節點與採取的行動。
        """
        g = self.build_graph(stm, ltm)
        try:
            nodes = nx.shortest_path(
                g,
                source=current,
                target=goal,
                weight=lambda u, v, d: 1 / d['weight'],
            )
        except nx.NetworkXNoPath:
            return None
        path = []
        for i in range(len(nodes) - 1):
            u = nodes[i]
            v = nodes[i + 1]
            path.append((v, g[u][v]['action']))
        return path
