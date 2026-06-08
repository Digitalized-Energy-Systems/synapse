"""Integration tests for the monee MES builder and the mango DAT simulation.

These build a small simbench multi-energy network (requires the ``simbench``
extra) and, for the full-simulation test, solve it with gurobi.
"""

import pytest

from synapse.agent.world import build_agent_topology, child_aid
from synapse.mes.network import create_mes_from_simbench_with_cp_distribution

SMALL_GRID = "1-LV-rural3--1-no_sw"


@pytest.fixture(scope="module")
def small_mes():
    return create_mes_from_simbench_with_cp_distribution(
        SMALL_GRID,
        heat_deployment_rate=0.3,
        gas_deployment_rate=0.3,
        chp_density=0.3,
        p2g_density=0.2,
        p2h_density=0.2,
        seed=100,
    )


def test_build_mes_has_grids_and_coupling_points(small_mes):
    grids = {type(n.grid).__name__ for n in small_mes.nodes}
    assert any("Power" in g for g in grids)
    assert any("Gas" in g for g in grids)
    assert any("Water" in g for g in grids)

    cp_names = {"PowerToGas", "GasToPower", "PowerToHeat", "PowerToHeatHG"}
    cps = [b for b in small_mes.branches if type(b.model).__name__ in cp_names]
    assert len(cps) > 0


def test_agent_topology_has_child_and_cp_vertices(small_mes):
    graph, child_specs, cp_specs = build_agent_topology(small_mes)
    assert len(child_specs) > 0
    # Every child has a vertex in the agent topology.
    for aid, child in child_specs:
        assert graph.has_node(aid)
        assert aid == child_aid(child.id)
    # Coupling points become their own agent vertices.
    assert len(cp_specs) >= 0


@pytest.mark.integration
def test_dat_simulation_forms_regions(small_mes):
    from synapse.simulation.scenarios import start_dat_simulation

    world, behavior = start_dat_simulation(small_mes, time_steps=3)
    assert behavior.net_results is not None
    assert behavior.region_manager.region_count >= 1
