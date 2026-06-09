"""Core coalition-formation primitives, ported to mango 2.x + monee.

* :class:`SecmesRegionManager` -- the coalition (region) registry.  Unchanged
  pure-networkx data structure; now owned by the environment behavior and shared
  across all agents (single-process simulation).
* :class:`SynapseAgentGraph` -- the agent topology used for neighborhood
  computation and the dynamic CP link/unlink of the DAT strategy.  Replaces the
  former ``SecmesAgentRouter``'s graph responsibilities; message dispatch is gone
  (agents now send real mango messages).
* :class:`SynapseRole` -- base role bound to the behavior, with mango send and a
  small gossip-cache helper used by the coalition roles.
"""

from __future__ import annotations

from typing import List, Set

import networkx as nx
from mango import Role

NETS_ACCESS = "nets"
VIRTUAL_NODE_CONTAIN_STR = ["node-", "junction", "bus", "virt"]


class SecmesRegionManager:
    """Coalition registry: a graph whose nodes are regions, each holding a set of
    assigned agent ids."""

    def __init__(self) -> None:
        self._region_graph = nx.Graph()

    def _generate_region_id(self):
        return (
            0
            if len(self._region_graph.nodes) == 0
            else max(self._region_graph.nodes.keys()) + 1
        )

    def register_region(self, neighbor_regions, initial_aid):
        region_id = self._generate_region_id()
        self._region_graph.add_node(region_id, assigned_agents={initial_aid})
        for region_neighbor in neighbor_regions:
            self._region_graph.add_edge(region_id, region_neighbor)
        return region_id

    def add_region(self, neighbor_regions, aid_set):
        region_id = self._generate_region_id()
        self._region_graph.add_node(region_id, assigned_agents=set(aid_set))
        for region_neighbor in neighbor_regions:
            self._region_graph.add_edge(region_id, region_neighbor)
        return region_id

    def register_agent(self, agent_id, region_id):
        # The target region may have been merged/removed by a concurrently
        # handled message before this (async) join is applied; ignore the stale
        # join rather than crashing -- the requester retries next round.
        if region_id not in self._region_graph.nodes:
            return
        for node in list(self._region_graph.nodes):
            assigned: Set = self._region_graph.nodes[node]["assigned_agents"]
            if agent_id in assigned:
                assigned.remove(agent_id)
                if not assigned:
                    self._region_graph.remove_node(node)
                    break
        self._region_graph.nodes[region_id]["assigned_agents"] |= {agent_id}

    def get_agent_region(self, agent_id):
        for node in self._region_graph.nodes:
            if agent_id in self._region_graph.nodes[node]["assigned_agents"]:
                return node
        return None

    def get_agents_region(self, region_id) -> Set:
        if region_id not in self._region_graph.nodes:
            return set()
        return self._region_graph.nodes[region_id]["assigned_agents"]

    def get_data_as_copy(self):
        region_graph_copy = self._region_graph.copy()
        for node in region_graph_copy.nodes:
            region_graph_copy.nodes[node]["assigned_agents"] = region_graph_copy.nodes[
                node
            ]["assigned_agents"].copy()
        return region_graph_copy

    def remove_assigned_agent(self, aid, region_id):
        self._region_graph.nodes[region_id]["assigned_agents"] -= {aid}

    def remove_region(self, region_id):
        self._region_graph.remove_node(region_id)

    def remove_own_region(self, agent_id):
        self.remove_region(self.get_agent_region(agent_id))

    @property
    def region_count(self) -> int:
        return len(self._region_graph.nodes)


def is_virtual_node(agent_id):
    for contain_str in VIRTUAL_NODE_CONTAIN_STR:
        if contain_str in str(agent_id):
            return True
    return False


def calc_k_nearest_neighbors_excluding_virtual(node_to_len_map, k, nodes_data):
    pairs: List = list(node_to_len_map.items())
    pairs.sort(key=lambda v: v[1])
    return [
        node
        for node, _ in pairs
        if not (
            is_virtual_node(node)
            or ("virt" in nodes_data[node] and nodes_data[node]["virt"])
        )
    ][:k]


class SynapseAgentGraph:
    """Agent topology over the monee network.

    Graph nodes are agent ids; passive monee nodes (buses / junctions) are kept as
    *virtual* connectors so neighborhoods route through them but never select them
    as cell agents.  Edge weights carry the physical loss / efficiency used for
    the distance-bounded neighborhood search.
    """

    def __init__(self, agent_topology: nx.Graph, neighborhood_size=10) -> None:
        self._agent_topology = agent_topology
        self._neighborhood_size = neighborhood_size
        self._edges_removed = {}
        self._subgraph_removed_cp = {}

    @property
    def topology(self):
        return self._agent_topology

    def exists(self, agent_id):
        return agent_id in self._agent_topology.nodes

    def unlink_cp(self, cp_id):
        """Cut a coupling-point agent out of the topology (DAT splitting)."""
        if cp_id not in self._agent_topology.nodes:
            return
        incident = list(self._agent_topology.edges(cp_id, keys=True)) if isinstance(
            self._agent_topology, nx.MultiGraph
        ) else [(u, v, None) for u, v in self._agent_topology.edges(cp_id)]
        self._edges_removed[cp_id] = []
        for u, v, key in incident:
            data = (
                self._agent_topology.edges[u, v, key]
                if key is not None
                else self._agent_topology.edges[u, v]
            )
            self._edges_removed[cp_id].append((u, v, key, dict(data)))
            if key is not None:
                self._agent_topology.remove_edge(u, v, key)
            else:
                self._agent_topology.remove_edge(u, v)
        self._agent_topology.nodes[cp_id]["virt"] = True

    def link_cp(self, cp_id):
        for u, v, key, data in self._edges_removed.get(cp_id, []):
            if key is not None:
                self._agent_topology.add_edge(u, v, key=key, **data)
            else:
                self._agent_topology.add_edge(u, v, **data)
        if cp_id in self._agent_topology.nodes:
            self._agent_topology.nodes[cp_id]["virt"] = False
        self._edges_removed.pop(cp_id, None)

    def calc_neighborhood(self, agent_id, cutoff_length=0.1):
        if agent_id not in self._agent_topology.nodes:
            return []
        lengths = nx.single_source_dijkstra_path_length(
            self._agent_topology, agent_id, cutoff=cutoff_length, weight="weight"
        )
        return calc_k_nearest_neighbors_excluding_virtual(
            lengths, self._neighborhood_size, self._agent_topology.nodes
        )

    def calc_cp_neighborhood(self, cp_id, networks=None, cutoff_length=0.1):
        # The CP is a single agent node bridging its grids; its neighborhood is
        # simply the distance-bounded neighborhood from that node.
        return self.calc_neighborhood(cp_id, cutoff_length=cutoff_length)

    def lookup_direct_neighbors(self, agent_id, include_virtual_nodes=False, blacklist=None):
        if agent_id not in self._agent_topology.nodes:
            return set()
        block = set(blacklist) if blacklist else set()
        direct = {
            n for n in nx.neighbors(self._agent_topology, agent_id) if n not in block
        }
        if include_virtual_nodes:
            return direct

        # Real-agent neighbours are reached by hopping through the *virtual*
        # connector nodes (``node-<id>``), which form a densely interconnected
        # mesh.  Expand each virtual node at most once via a shared visited set;
        # the previous per-branch blacklist (passed by value) re-expanded shared
        # virtual nodes through exponentially many paths and never terminated on
        # real grids (only the CONNECTED_COMPONENTS splitting path hit this).
        virtual = {n for n in direct if is_virtual_node(n)}
        result = direct - virtual
        expanded = set(block)
        expanded.update(virtual)
        frontier = list(virtual)
        while frontier:
            vn = frontier.pop()
            for n in nx.neighbors(self._agent_topology, vn):
                if n in expanded:
                    continue
                if is_virtual_node(n):
                    expanded.add(n)
                    frontier.append(n)
                else:
                    result.add(n)
        return result

    def get_agents_as_subgraph(self, agent_ids: List[str]):
        return self._agent_topology.subgraph(agent_ids)

    def get_data_as_copy(self):
        copy = self._agent_topology.copy()
        for node_key in copy.nodes:
            copy.nodes[node_key]["agent"] = None
            copy.nodes[node_key]["roles"] = None
        return copy

    def get_data_as_ref(self):
        return self._agent_topology


class SynapseRole(Role):
    """Base role bound to the :class:`SynapseEnvironmentBehavior`.

    Provides mango message sending (aid -> address via the behavior) and shortcut
    access to the shared region manager / agent graph.
    """

    def __init__(self, behavior) -> None:
        super().__init__()
        self._behavior = behavior

    @property
    def behavior(self):
        return self._behavior

    @property
    def region_manager(self) -> SecmesRegionManager:
        return self._behavior.region_manager

    @property
    def graph(self) -> SynapseAgentGraph:
        return self._behavior.agent_graph

    @property
    def aid(self):
        return self.context.aid

    async def send(self, target_aid, content) -> bool:
        addr = self._behavior.address_of(target_aid)
        if addr is None:
            return False
        return await self.context.send_message(content, addr)
