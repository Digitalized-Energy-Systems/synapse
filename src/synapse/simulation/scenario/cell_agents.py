"""Cell-simulation entry points on the mango + monee stack."""

from __future__ import annotations

import asyncio

from synapse.agent.cell_agent import CellAgentRole
from synapse.agent.world import (
    build_world,
    register_default_recordings,
    run_world,
)


def default_cell_role_factory(behavior, aid, ca, is_cp, networks):
    return CellAgentRole(behavior, ca, is_cp=is_cp, networks=networks)


def start_cell_simulation(
    net,
    cell_role_factory=default_cell_role_factory,
    cp_extra_roles_factory=None,
    demand_schedule=None,
    rounds: int = 96,
    solver: str = "gurobi",
    neighborhood_size: int = 10,
):
    """Build and run a cell-style coalition simulation; returns
    ``(world, behavior)`` with the final region manager and energy-flow results.

    Agents must be registered with a running event loop (mango starts each
    agent's inbox task on registration), so the whole build+run happens inside a
    single :func:`asyncio.run`.
    """

    async def _build_and_run():
        world, behavior = build_world(
            net,
            cell_role_factory=cell_role_factory,
            cp_extra_roles_factory=cp_extra_roles_factory,
            solver=solver,
            demand_schedule=demand_schedule,
            neighborhood_size=neighborhood_size,
        )
        register_default_recordings(world, behavior)
        await run_world(world, behavior, rounds=rounds)
        return world, behavior

    return asyncio.run(_build_and_run())
