"""Simulation scenarios for the dynamic adaptive topology (DAT) experiment."""

from __future__ import annotations

from synapse.agent.dat import (
    DATCouplingPointRole,
    DynamicCoalitionAdaptionTopologyAgent,
    SplittingStrategy,
)
from synapse.simulation.profiles import random_demand_schedule
from synapse.simulation.scenario.cell_agents import start_cell_simulation

CP_CHANGE_PROB = 0.2
TIME_STEPS = 96
DAT_SIM_NAME = "DATSIM"


def start_dat_simulation(
    net,
    cp_change_prob: float = CP_CHANGE_PROB,
    splitting_strategy: SplittingStrategy = SplittingStrategy.DISINTEGRATE,
    time_steps: int = TIME_STEPS,
    demand_schedule=None,
    attach_random_demand: bool = True,
    solver: str = "gurobi",
):
    """Run the DAT coalition-formation scenario: every cell agent is a
    :class:`DynamicCoalitionAdaptionTopologyAgent`, and every coupling point also
    carries a :class:`DATCouplingPointRole` that toggles it on/off, splitting its
    region per ``splitting_strategy``."""
    if demand_schedule is None and attach_random_demand:
        demand_schedule = random_demand_schedule(net, time_steps, seed=100)

    def cell_role_factory(behavior, aid, ca, is_cp, networks):
        return DynamicCoalitionAdaptionTopologyAgent(
            behavior, ca, is_cp=is_cp, networks=networks
        )

    def cp_extra_roles_factory(behavior, aid, networks):
        return [DATCouplingPointRole(behavior, networks, cp_change_prob, splitting_strategy)]

    return start_cell_simulation(
        net,
        cell_role_factory=cell_role_factory,
        cp_extra_roles_factory=cp_extra_roles_factory,
        demand_schedule=demand_schedule,
        rounds=time_steps,
        solver=solver,
    )
