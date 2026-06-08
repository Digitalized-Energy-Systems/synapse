"""Dynamic Adaptive Topology (DAT) roles, ported to mango 2.x + monee.

A coupling point periodically toggles itself on/off; when it switches off it is
cut out of the agent topology and its region is split according to the chosen
:class:`SplittingStrategy`.  The physical on/off is applied through monee via the
behavior's ``regulate`` action.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import List

import networkx as nx
from overrides import overrides

from synapse.agent.cell_agent import CellAgentRole
from synapse.agent.core import SecmesRegionManager, SynapseAgentGraph, SynapseRole

CP_TOGGLE_PERIOD_S = 1.0


@dataclass
class RegionDisbandedMessage:
    sender_aid: str
    region_id: int
    time: int


class SplittingStrategy(Enum):
    CONNECTED_COMPONENTS = 1
    DISINTEGRATE = 2


async def execute_splitting_strategy(
    role: SynapseRole,
    strategy: SplittingStrategy,
    graph: SynapseAgentGraph,
    region_manager: SecmesRegionManager,
    cp_aid: str,
    networks: List[str],
    time: int,
):
    region = region_manager.get_agent_region(cp_aid)
    if region is None:
        graph.unlink_cp(cp_aid)
        return
    agents_in_own_region = set(region_manager.get_agents_region(region))
    agents_in_own_region.discard(cp_aid)

    if strategy == SplittingStrategy.DISINTEGRATE:
        graph.unlink_cp(cp_aid)
        subgraph = graph.get_agents_as_subgraph(list(agents_in_own_region))
        if len(list(nx.connected_components(subgraph))) <= 1:
            region_manager.remove_assigned_agent(cp_aid, region)
        else:
            region_manager.remove_region(region)
            for aid in agents_in_own_region:
                await role.send(aid, RegionDisbandedMessage(cp_aid, region, time))
    else:  # CONNECTED_COMPONENTS
        region_manager.remove_region(region)
        graph.unlink_cp(cp_aid)
        subgraph = graph.get_agents_as_subgraph(list(agents_in_own_region))
        for component in nx.connected_components(subgraph):
            neighbor_regions = set()
            for node in component:
                for neighbor in graph.lookup_direct_neighbors(node):
                    nr = region_manager.get_agent_region(neighbor)
                    if nr is not None:
                        neighbor_regions |= {nr}
            region_manager.add_region(neighbor_regions, component)


class DynamicCoalitionAdaptionTopologyAgent(CellAgentRole):
    """A :class:`CellAgentRole` that reacts to region disbanding and does not
    self-adjust its operation point (topology adaption is the driver here)."""

    @overrides
    def setup(self):
        super().setup()
        self.context.subscribe_message(
            self,
            self.handle_region_disbanded,
            lambda c, _: isinstance(c, RegionDisbandedMessage),
        )

    def handle_region_disbanded(self, msg: RegionDisbandedMessage, _meta):
        # Re-bootstrap a region for this agent on the next opportunity.
        self.create_initial_region()

    @overrides
    def execute_operation_point(self, time):
        # No operation-point adjustment in the DAT scenario.
        pass


class DATCouplingPointRole(SynapseRole):
    """Periodically toggles a coupling point, splitting its region when it goes
    offline and re-forming one when it comes back."""

    def __init__(
        self,
        behavior,
        cp_aid_networks,
        probability: float,
        splitting_strategy: SplittingStrategy = SplittingStrategy.DISINTEGRATE,
    ) -> None:
        super().__init__(behavior)
        self._networks = cp_aid_networks
        self._toggle = True
        self._probability = probability
        self._splitting_strategy = splitting_strategy
        self._round = 0

    def setup(self):
        self.context.schedule_periodic_task(self._toggle_round, CP_TOGGLE_PERIOD_S)

    async def _toggle_round(self):
        time = self._round
        self._round += 1
        if time <= 2:
            return
        if random.random() >= self._probability:
            return

        region_m: SecmesRegionManager = self.region_manager
        graph: SynapseAgentGraph = self.graph
        self._toggle = not self._toggle
        if self._toggle:
            # Switch the coupling point back on.
            if self.behavior.has_action(self.aid, "regulate"):
                self.behavior.act(self.aid, "regulate", 1.0)
            graph.link_cp(self.aid)
            region_m.register_region(set(), self.aid)
        else:
            # Shut it down physically and split its region.
            if self.behavior.has_action(self.aid, "regulate"):
                self.behavior.act(self.aid, "regulate", 0.0)
            await execute_splitting_strategy(
                self,
                self._splitting_strategy,
                graph,
                region_m,
                self.aid,
                self._networks,
                time,
            )
