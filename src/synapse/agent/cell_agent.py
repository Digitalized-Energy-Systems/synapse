"""Cell agents for coalition formation, ported to mango 2.x + monee.

The coalition-formation algorithm (attraction-based region joining + a local
distributed-gradient operation point) is preserved, but the former *synchronous*
in-process round-trips are replaced by genuine mango message passing.  Because a
blocking ``await reply`` would not advance the discrete-event clock, the protocol
is gossip / event driven: agents publish their balance to neighbors, cache what
they hear, and act each control round on the cached view (which converges over
rounds).  Physical quantities are read from monee via ``behavior.observe(aid)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from synapse.agent.core import SecmesRegionManager, SynapseAgentGraph, SynapseRole
from synapse.da.dgd import (
    generate_random_doubly_stoch_mat,
    iteration_step,
)

CONTROL_PERIOD_S = 1.0


# --------------------------------------------------------------------- messages
@dataclass
class AgentBalanceAnnounce:
    sender_aid: str
    balance: list


@dataclass
class JoinRequest:
    sender_aid: str
    region_id: int
    region_attraction: list


# --------------------------------------------------------- component access layer
def to_multi_energy(power=0.0, heat=0.0, gas=0.0):
    return np.array([float(power), float(heat), float(gas)])


def _f(obs, key, default=0.0):
    val = obs.get(key, default)
    if hasattr(val, "value"):
        val = val.value
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


class CommonNetworkComponentAccess(ABC):
    """Reads a component's multi-energy balance / capability from monee results.

    Balance uses the *injection* convention (positive = surplus into the grid):
    monee stores demands in load convention (load/sink/heat-load positive), so
    the injection balance is simply the negated observed setpoint.
    """

    def __init__(self, behavior, aid) -> None:
        self._behavior = behavior
        self._aid = aid

    def _obs(self):
        return self._behavior.observe(self._aid)

    @abstractmethod
    def calc_balance(self):
        ...

    @abstractmethod
    def max_energy(self):
        ...

    def define_local_constraints(self):
        return lambda r: None


class PowerCA(CommonNetworkComponentAccess):
    def calc_balance(self):
        obs = self._obs()
        return to_multi_energy(power=-_f(obs, "p_mw"))

    def max_energy(self):
        return to_multi_energy(power=abs(_f(self._obs(), "p_mw")))


class GasCA(CommonNetworkComponentAccess):
    def calc_balance(self):
        obs = self._obs()
        return to_multi_energy(gas=-_f(obs, "mass_flow"))

    def max_energy(self):
        return to_multi_energy(gas=abs(_f(self._obs(), "mass_flow")))


class HeatCA(CommonNetworkComponentAccess):
    def calc_balance(self):
        obs = self._obs()
        return to_multi_energy(heat=-_f(obs, "q_mw_heat"))

    def max_energy(self):
        return to_multi_energy(heat=abs(_f(self._obs(), "q_mw_heat")))


class DummyHeatCA(CommonNetworkComponentAccess):
    def calc_balance(self):
        return to_multi_energy()

    def max_energy(self):
        return to_multi_energy()


class PowerGasCA(CommonNetworkComponentAccess):
    def calc_balance(self):
        obs = self._obs()
        return to_multi_energy(power=-_f(obs, "p_mw"), gas=-_f(obs, "mass_flow"))

    def max_energy(self):
        obs = self._obs()
        return to_multi_energy(
            power=abs(_f(obs, "p_mw")), gas=abs(_f(obs, "mass_flow"))
        )


class PowerHeatCA(CommonNetworkComponentAccess):
    def calc_balance(self):
        obs = self._obs()
        return to_multi_energy(power=-_f(obs, "p_mw"), heat=-_f(obs, "q_mw_heat"))

    def max_energy(self):
        obs = self._obs()
        return to_multi_energy(
            power=abs(_f(obs, "p_mw")), heat=abs(_f(obs, "q_mw_heat"))
        )


class PowerGasHeatCA(CommonNetworkComponentAccess):
    def calc_balance(self):
        obs = self._obs()
        return to_multi_energy(
            power=-_f(obs, "p_mw"),
            heat=-_f(obs, "q_mw_heat"),
            gas=-_f(obs, "mass_flow"),
        )

    def max_energy(self):
        obs = self._obs()
        return to_multi_energy(
            power=abs(_f(obs, "p_mw")),
            heat=abs(_f(obs, "q_mw_heat")),
            gas=abs(_f(obs, "mass_flow")),
        )


class CouplingPointCA(CommonNetworkComponentAccess):
    """Balance/capability of a coupling-point branch read from monee branch
    results.  Branch result dicts expose the per-carrier flows under their
    ``*_from_mw`` / ``mass_flow`` / ``q_mw_heat`` keys (scaled by ``regulation``)."""

    def _carrier_flows(self):
        obs = self._obs()
        reg = _f(obs, "regulation", 1.0)
        power = _f(obs, "p_from_mw", _f(obs, "p_mw")) * reg
        gas = _f(obs, "mass_flow") * reg
        heat = _f(obs, "q_mw_heat") * reg
        return power, gas, heat

    def calc_balance(self):
        power, gas, heat = self._carrier_flows()
        return to_multi_energy(power=-power, heat=-heat, gas=-gas)

    def max_energy(self):
        power, gas, heat = self._carrier_flows()
        return to_multi_energy(power=abs(power), heat=abs(heat), gas=abs(gas))


# --------------------------------------------------------------------- the agent
class CellAgentRole(SynapseRole, ABC):
    """Attraction-based coalition-formation agent.

    Each control round the agent publishes its balance, refreshes its view of the
    region balance from cached neighbour balances, and asks more-attractive
    neighbours to merge regions.  The shared :class:`SecmesRegionManager` keeps
    the coalition state consistent across the single-process simulation.
    """

    def __init__(self, behavior, common_nc_data_access, is_cp=False, networks=None):
        super().__init__(behavior)
        self._ca = common_nc_data_access
        self._is_cp = is_cp
        self._networks = networks or []
        self._peer_balance = {}
        self._operation_point = {}
        self._round = 0
        self.last_control_values = {}

    # ----- mango lifecycle
    def setup(self):
        self.context.subscribe_message(
            self, self.handle_join_request, lambda c, _: isinstance(c, JoinRequest)
        )
        self.context.subscribe_message(
            self,
            self.handle_balance_announce,
            lambda c, _: isinstance(c, AgentBalanceAnnounce),
        )
        self.create_initial_region()
        self.context.schedule_periodic_task(self._control_round, CONTROL_PERIOD_S)

    # ----- region bootstrap
    def create_initial_region(self):
        region_m: SecmesRegionManager = self.region_manager
        graph: SynapseAgentGraph = self.graph
        if graph.exists(self.aid) and region_m.get_agent_region(self.aid) is None:
            neighbors = self._neighbors()
            neighbor_regions = {
                region_m.get_agent_region(n)
                for n in neighbors
                if region_m.get_agent_region(n) is not None
            }
            return region_m.register_region(neighbor_regions, self.aid)
        return None

    def _neighbors(self):
        if self._is_cp:
            return self.graph.calc_cp_neighborhood(self.aid, self._networks)
        return self.graph.calc_neighborhood(self.aid)

    # ----- gossip handlers
    def handle_balance_announce(self, content: AgentBalanceAnnounce, _meta):
        self._peer_balance[content.sender_aid] = np.array(content.balance)

    def handle_join_request(self, content: JoinRequest, _meta):
        region_m = self.region_manager
        region = region_m.get_agent_region(self.aid)
        if region is None:
            return
        own_region_balance = self.calc_region_balance(region_m.get_agents_region(region))
        attraction_to_requester = self.calc_agent_attraction(
            content.sender_aid, own_region_balance
        )
        attraction_to_own = self.calc_agent_attraction(self.aid, self._ca.calc_balance())
        if (attraction_to_requester >= attraction_to_own).all():
            region_m.register_agent(self.aid, content.region_id)

    # ----- balances / attraction
    def _balance_of(self, agent_id):
        if agent_id == self.aid:
            return self._ca.calc_balance()
        return self._peer_balance.get(agent_id, to_multi_energy())

    def calc_region_balance(self, region_agents):
        total = to_multi_energy()
        for agent in region_agents:
            total = total + self._balance_of(agent)
        return total

    def calc_agent_attraction(self, other, calculated_balance):
        region = self.region_manager.get_agent_region(other)
        if region is None:
            other_balance = self._balance_of(other)
        else:
            other_balance = self.calc_region_balance(
                self.region_manager.get_agents_region(region)
            )
        return -np.sign(calculated_balance) * other_balance - self.calc_cost_gradient(other)

    def calc_cost_gradient(self, neighbor):
        return to_multi_energy()

    # ----- operation point (local distributed gradient step)
    def calculate_operation_point(self, time):
        if time in self._operation_point:
            return self._operation_point[time]
        region = self.region_manager.get_agent_region(self.aid)
        if region is None:
            return np.ones(3)
        region_agents = list(self.region_manager.get_agents_region(region))
        m = len(region_agents)
        i = region_agents.index(self.aid) if self.aid in region_agents else 0
        x = np.array([np.ones(3) for _ in range(m)])
        w = generate_random_doubly_stoch_mat((m, m))
        for j in range(20):
            new_x = iteration_step(
                w, x, lambda _x: self._ca.max_energy(), m, i, j
            )
            x[i] = new_x
        self._operation_point[time] = x[i]
        return x[i]

    def execute_operation_point(self, time):
        op = self.calculate_operation_point(time)
        factor = float(np.clip(op[0], 0.0, 1.0))
        if self.behavior.has_action(self.aid, "regulate"):
            self.behavior.act(self.aid, "regulate", factor)

    # ----- per-round control
    async def _control_round(self):
        time = self._round
        self._round += 1

        region_m = self.region_manager
        region = region_m.get_agent_region(self.aid)
        if region is None:
            region = self.create_initial_region()
        if region is None:
            return

        own_balance = self._ca.calc_balance()
        neighbors = self._neighbors()

        # Publish our balance to neighbors so they can refresh their view.
        for neighbor in neighbors:
            await self.send(neighbor, AgentBalanceAnnounce(self.aid, own_balance.tolist()))

        region_agents = region_m.get_agents_region(region)
        region_balance = self.calc_region_balance(region_agents)

        control_values = {
            "region_balance_power": float(region_balance[0]),
            "region_balance_heat": float(region_balance[1]),
            "region_balance_gas": float(region_balance[2]),
        }

        for neighbor in neighbors:
            attraction = self.calc_agent_attraction(neighbor, region_balance)
            if neighbor in region_agents:
                continue
            if (attraction >= 0).all():
                await self.send(
                    neighbor, JoinRequest(self.aid, region, attraction.tolist())
                )

        self.execute_operation_point(time)

        current_region = region_m.get_agent_region(self.aid)
        control_values["region"] = -1 if current_region is None else current_region
        self.last_control_values = control_values
        return control_values
