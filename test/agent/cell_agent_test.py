"""Fast, solver-free unit tests for the coalition primitives."""

from types import SimpleNamespace

import networkx as nx
import numpy as np

from synapse.agent.cell_agent import CellAgentRole, to_multi_energy
from synapse.agent.core import SecmesRegionManager, SynapseAgentGraph


class _StubBehavior:
    def __init__(self, region_manager, agent_graph):
        self.region_manager = region_manager
        self.agent_graph = agent_graph
        self.aid_to_addr = {}

    def address_of(self, aid):
        return self.aid_to_addr.get(aid)

    def observe(self, aid):
        return {}

    def has_action(self, aid, action):
        return False


class _ConstCA:
    def __init__(self, balance):
        self._balance = np.array(balance, dtype=float)

    def calc_balance(self):
        return self._balance

    def max_energy(self):
        return np.abs(self._balance)


def _make_agent(own_balance, aid="self"):
    rm = SecmesRegionManager()
    graph = SynapseAgentGraph(nx.Graph())
    behavior = _StubBehavior(rm, graph)
    agent = CellAgentRole(behavior, _ConstCA(own_balance))
    # Bind a minimal context so ``agent.aid`` resolves without a mango container.
    agent._context = SimpleNamespace(aid=aid)
    return agent, rm


def test_region_manager_register_and_merge():
    rm = SecmesRegionManager()
    r0 = rm.register_region(set(), "a")
    rm.register_region(set(), "b")
    assert rm.region_count == 2
    rm.register_agent("b", r0)  # move b into a's region
    assert rm.get_agent_region("a") == r0
    assert rm.get_agent_region("b") == r0
    assert rm.region_count == 1  # b's old (now empty) region removed


def test_attraction_surplus_attracts_deficit():
    # An agent with a power deficit is attracted to a neighbour with surplus.
    agent, _ = _make_agent(own_balance=[-200, 0, 0])
    agent._peer_balance["other"] = to_multi_energy(power=200)
    attraction = agent.calc_agent_attraction("other", np.array([-200.0, 0, 0]))
    assert (attraction >= 0).all()


def test_attraction_same_sign_repels():
    agent, _ = _make_agent(own_balance=[200, 0, 0])
    agent._peer_balance["other"] = to_multi_energy(power=200)
    attraction = agent.calc_agent_attraction("other", np.array([200.0, 0, 0]))
    assert (attraction <= 0).all()


def test_region_balance_sums_cached_peers():
    agent, rm = _make_agent(own_balance=[5, 0, 0], aid="self")
    agent._peer_balance["p1"] = to_multi_energy(power=3)
    total = agent.calc_region_balance({"self", "p1"})
    assert np.isclose(total[0], 8.0)
