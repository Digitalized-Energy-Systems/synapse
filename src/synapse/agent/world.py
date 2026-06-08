"""Mango simulation world for synapse, built on monee physics.

Replaces the former pandapipes/pandapower driven ``AsyncWorld`` / ``SyncWorld``.
A :class:`~mango.simulation.world.SimulationWorld` is assembled with a
:class:`~synapse.agent.behavior.SynapseEnvironmentBehavior`; one mango
``RoleAgent`` is created per monee child and per coupling-point branch, the
coalition agent topology is derived from the physical network, and the world is
advanced with ``discrete_step_until``.
"""

from __future__ import annotations

import networkx as nx
from mango import RoleAgent, create_world, discrete_step_until, record_world
from mango.simulation.communication import SimpleCommunicationSimulation
from mango.simulation.environment import DefaultEnvironment

from synapse.agent.behavior import SynapseEnvironmentBehavior
from synapse.agent.cell_agent import (
    CouplingPointCA,
    DummyHeatCA,
    GasCA,
    HeatCA,
    PowerCA,
    CONTROL_PERIOD_S,
)
from synapse.agent.core import SecmesRegionManager, SynapseAgentGraph

CP_MODEL_NAMES = {"PowerToGas", "GasToPower", "PowerToHeat", "PowerToHeatHG"}
PHYS_EDGE_WEIGHT = 0.02


def node_graph_aid(node_id):
    return f"node-{node_id}"


def child_aid(child_id):
    return f"child-{child_id}"


def cp_aid(branch_id):
    return f"cp-{branch_id[0]}-{branch_id[1]}-{branch_id[2]}"


def _grids_of(net, *node_ids):
    grids = []
    for nid in node_ids:
        grid = getattr(net.node_by_id(nid), "grid", None)
        name = getattr(grid, "name", None) or (
            type(grid).__name__ if grid is not None else None
        )
        if name is not None:
            grids.append(name)
    return sorted(set(grids))


def build_agent_topology(net):
    """Project the monee network onto the coalition agent topology.

    Returns ``(graph, child_specs, cp_specs)`` where ``child_specs`` is a list of
    ``(aid, child)`` and ``cp_specs`` a list of ``(aid, branch_id, networks)``.
    Monee nodes become *virtual* connector vertices (``node-<id>``); childs and
    coupling points become agent vertices.
    """
    graph = nx.Graph()
    for node in net.nodes:
        graph.add_node(node_graph_aid(node.id), virt=True)

    child_specs = []
    for child in net.childs:
        aid = child_aid(child.id)
        graph.add_node(aid, virt=False)
        graph.add_edge(aid, node_graph_aid(child.node_id), weight=0.0)
        child_specs.append((aid, child))

    cp_specs = []
    for branch in net.branches:
        u, v = branch.id[0], branch.id[1]
        mtype = type(branch.model).__name__
        if mtype in CP_MODEL_NAMES:
            aid = cp_aid(branch.id)
            graph.add_node(aid, virt=False)
            graph.add_edge(aid, node_graph_aid(u), weight=0.0)
            graph.add_edge(aid, node_graph_aid(v), weight=0.0)
            cp_specs.append((aid, branch.id, _grids_of(net, u, v)))
        else:
            if graph.has_node(node_graph_aid(u)) and graph.has_node(node_graph_aid(v)):
                graph.add_edge(
                    node_graph_aid(u), node_graph_aid(v), weight=PHYS_EDGE_WEIGHT
                )
    return graph, child_specs, cp_specs


def ca_for_child(behavior, aid, child, net):
    node = net.node_by_id(child.node_id)
    grid_name = (type(node.grid).__name__ if node.grid is not None else "").lower()
    model_name = type(child.model).__name__
    if model_name in ("HeatLoad", "HeatGenerator"):
        return HeatCA(behavior, aid)
    if "power" in grid_name:
        return PowerCA(behavior, aid)
    if "gas" in grid_name:
        return GasCA(behavior, aid)
    if "water" in grid_name:
        return DummyHeatCA(behavior, aid)
    return PowerCA(behavior, aid)


def build_world(
    net,
    *,
    cell_role_factory,
    cp_extra_roles_factory=None,
    solver="gurobi",
    demand_schedule=None,
    neighborhood_size=10,
    static_delay_s=0.02,
):
    """Assemble the mango simulation world for ``net``.

    ``cell_role_factory(behavior, aid, ca, is_cp, networks)`` returns the primary
    cell role for an agent; ``cp_extra_roles_factory(behavior, aid, networks)``
    (optional) returns extra roles to add to coupling-point agents (e.g. the DAT
    toggling role).  Returns ``(world, behavior)``.
    """
    graph, child_specs, cp_specs = build_agent_topology(net)
    region_manager = SecmesRegionManager()
    agent_graph = SynapseAgentGraph(graph, neighborhood_size)
    behavior = SynapseEnvironmentBehavior(
        net,
        solver=solver,
        demand_schedule=demand_schedule,
        region_manager=region_manager,
        agent_graph=agent_graph,
    )
    env = DefaultEnvironment(behavior=behavior)
    com = SimpleCommunicationSimulation(default_delay_s=static_delay_s)
    world = create_world(start_time=0.0, communication_sim=com, environment=env)

    for aid, child in child_specs:
        ca = ca_for_child(behavior, aid, child, net)
        role = cell_role_factory(behavior, aid, ca, False, [])
        agent = world.register(RoleAgent(), suggested_aid=aid)
        agent.add_role(role)
        behavior.install(agent, id=child.id, type="child")

    for aid, branch_id, networks in cp_specs:
        ca = CouplingPointCA(behavior, aid)
        role = cell_role_factory(behavior, aid, ca, True, networks)
        agent = world.register(RoleAgent(), suggested_aid=aid)
        agent.add_role(role)
        behavior.cp_networks[aid] = networks
        if cp_extra_roles_factory is not None:
            for extra in cp_extra_roles_factory(behavior, aid, networks):
                agent.add_role(extra)
        behavior.install(agent, id=branch_id, type="branch")

    for aid, agent in world._agents.items():
        behavior.aid_to_addr[aid] = agent.addr

    return world, behavior


def register_default_recordings(world, behavior):
    record_world(world, "region_count", lambda: behavior.region_manager.region_count)
    record_world(world, "step_index", lambda: behavior.step_index)


async def run_world(world, behavior, rounds=10, control_period_s=CONTROL_PERIOD_S):
    """Run the world for approximately ``rounds`` control rounds."""
    async with world:
        await discrete_step_until(
            world, max_advance_time_s=rounds * control_period_s + control_period_s
        )
    behavior.flush_energy_flow()
    return behavior
