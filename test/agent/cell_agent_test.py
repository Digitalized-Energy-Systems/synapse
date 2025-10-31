import peext.scenario.network as pn
import pandapipes.multinet.control as ppmc

from peext.node import (
    PowerLoadNode,
    GeneratorNode,
    HeatExchangerNode,
    SourceNode,
    CHPNode,
    P2GNode,
)
from mango.role.core import RoleAgentContext, RoleHandler
import numpy as np

from synapse.agent.cell_agent import (
    CellAgentRole,
    GasCA,
    HeatCA,
    PowerCA,
    PowerGasCA,
    PowerGasHeatCA,
)

from synapse.agent.core import SecmesRegionManager
from synapse.agent.core import SecmesAgentRouter

import networkx as nx


def test_attraction_id_load():
    # GIVEN
    mn = pn.create_small_test_multinet()
    load_model = PowerLoadNode(0, mn["nets"]["power"])
    ppmc.run_control_multinet.run_control(mn, max_iter=30, mode="all")
    load_cell_agent = CellAgentRole(load_model, PowerCA(load_model))
    region_m = SecmesRegionManager()
    region_m.register_region([], 0)
    load_cell_agent.secmes_setup(region_manager=region_m, sync_router=None)
    load_cell_agent.bind(RoleAgentContext(None, None, 0, None, None))

    # WHEN
    attraction = load_cell_agent.calc_agent_attraction(0, np.array([-200, 0, 0]))

    # THEN
    assert (attraction <= 0).all()


def test_attraction_id_generator():
    # GIVEN
    mn = pn.create_small_test_multinet()
    generator_model = GeneratorNode(0, mn["nets"]["power"])
    ppmc.run_control_multinet.run_control(mn, max_iter=30, mode="all")
    generator_agent = CellAgentRole(generator_model, PowerCA(generator_model))
    region_m = SecmesRegionManager()
    region_m.register_region([], 0)
    generator_agent.secmes_setup(region_manager=region_m, sync_router=None)
    generator_agent.bind(RoleAgentContext(None, None, 0, None, None))

    # WHEN
    attraction = generator_agent.calc_agent_attraction(0, np.array([200, 0, 0]))

    # THEN
    assert (attraction <= 0).all()


def test_attraction_generator_load():
    # GIVEN
    mn = pn.create_small_test_multinet()
    generator_model = GeneratorNode(0, mn["nets"]["power"])
    ppmc.run_control_multinet.run_control(mn, max_iter=30, mode="all")
    generator_agent = CellAgentRole(generator_model, PowerCA(generator_model))
    region_m = SecmesRegionManager()
    region_m.register_region([], 0)
    generator_agent.secmes_setup(region_manager=region_m, sync_router=None)
    generator_agent.bind(RoleAgentContext(None, None, 0, None, None))

    # WHEN
    attraction = generator_agent.calc_agent_attraction(0, np.array([-200, 0, 0]))

    # THEN
    assert (attraction >= 0).all()


def test_attraction_generator_load_small():
    # GIVEN
    mn = pn.create_small_test_multinet()
    generator_model = GeneratorNode(0, mn["nets"]["power"])
    ppmc.run_control_multinet.run_control(mn, max_iter=30, mode="all")
    generator_agent = CellAgentRole(generator_model, PowerCA(generator_model))
    region_m = SecmesRegionManager()
    region_m.register_region([], 0)
    generator_agent.secmes_setup(region_manager=region_m, sync_router=None)
    generator_agent.bind(RoleAgentContext(None, None, 0, None, None))

    # WHEN
    attraction = generator_agent.calc_agent_attraction(0, np.array([-0.0001, 0, 0]))

    # THEN
    assert (attraction >= 0).all()


def test_attraction_heat_exchanger_id():
    # GIVEN
    mn = pn.create_small_test_multinet()
    he_model = HeatExchangerNode(0, mn["nets"]["heat"])
    ppmc.run_control_multinet.run_control(mn, max_iter=30, mode="all")
    he_agent = CellAgentRole(he_model, HeatCA(he_model))
    region_m = SecmesRegionManager()
    region_m.register_region([], 0)
    he_agent.secmes_setup(region_manager=region_m, sync_router=None)
    he_agent.bind(RoleAgentContext(None, None, 0, None, None))

    # WHEN
    attraction = he_agent.calc_agent_attraction(0, np.array([0, -20, 0]))

    # THEN
    assert (attraction <= 0).all()


def test_attraction_heat_exchanger_diff():
    # GIVEN
    mn = pn.create_small_test_multinet()
    he_model = HeatExchangerNode(0, mn["nets"]["heat"])
    ppmc.run_control_multinet.run_control(mn, max_iter=30, mode="all")
    he_agent = CellAgentRole(he_model, HeatCA(he_model))
    region_m = SecmesRegionManager()
    region_m.register_region([], 0)
    he_agent.secmes_setup(region_manager=region_m, sync_router=None)
    he_agent.bind(RoleAgentContext(None, None, 0, None, None))

    # WHEN
    attraction = he_agent.calc_agent_attraction(0, np.array([0, 20, 0]))

    # THEN
    assert (attraction >= 0).all()


def test_attraction_gas_id():
    # GIVEN
    mn = pn.create_small_test_multinet()
    model = SourceNode(0, mn["nets"]["gas"])
    ppmc.run_control_multinet.run_control(mn, max_iter=30, mode="all")
    agent = CellAgentRole(model, GasCA(model))
    region_m = SecmesRegionManager()
    region_m.register_region([], 0)
    agent.secmes_setup(region_manager=region_m, sync_router=None)
    agent.bind(RoleAgentContext(None, None, 0, None, None))

    # WHEN
    attraction = agent.calc_agent_attraction(0, np.array([0, 0, 20]))

    # THEN
    assert (attraction <= 0).all()


def test_attraction_gas_diff():
    # GIVEN
    mn = pn.create_small_test_multinet()
    model = SourceNode(0, mn["nets"]["gas"])
    ppmc.run_control_multinet.run_control(mn, max_iter=30, mode="all")
    agent = CellAgentRole(model, GasCA(model))
    region_m = SecmesRegionManager()
    region_m.register_region([], 0)
    agent.secmes_setup(region_manager=region_m, sync_router=None)
    agent.bind(RoleAgentContext(None, None, 0, None, None))

    # WHEN
    attraction = agent.calc_agent_attraction(0, np.array([0, 0, -20]))

    # THEN
    assert (attraction >= 0).all()


def test_attraction_chp_full_diff():
    # GIVEN
    mn = pn.create_small_test_multinet()
    model = CHPNode(0, mn)
    ppmc.run_control_multinet.run_control(mn, max_iter=30, mode="all")
    agent = CellAgentRole(model, GasCA(model))
    region_m = SecmesRegionManager()
    region_m.register_region([], 0)
    agent.secmes_setup(region_manager=region_m, sync_router=None)
    agent.bind(RoleAgentContext(None, None, 0, None, None))

    # WHEN
    attraction = agent.calc_agent_attraction(0, np.array([-10, -10, 10]))

    # THEN
    assert (attraction >= 0).all()


def test_attraction_chp_same_cases():
    # GIVEN
    mn = pn.create_small_test_multinet()
    model = CHPNode(0, mn)
    ppmc.run_control_multinet.run_control(mn, max_iter=30, mode="all")
    agent = CellAgentRole(model, GasCA(model))
    region_m = SecmesRegionManager()
    region_m.register_region([], 0)
    agent.secmes_setup(region_manager=region_m, sync_router=None)
    agent.bind(RoleAgentContext(None, None, 0, None, None))

    # WHEN
    attraction = agent.calc_agent_attraction(0, np.array([10, 10, -10]))
    attraction2 = agent.calc_agent_attraction(0, np.array([0, 10, 0]))
    attraction3 = agent.calc_agent_attraction(0, np.array([10, 0, 0]))
    attraction4 = agent.calc_agent_attraction(0, np.array([0, 0, -10]))

    # THEN
    assert (attraction <= 0).all()
    assert (attraction2 <= 0).all()
    assert (attraction3 <= 0).all()
    assert (attraction4 <= 0).all()


def test_mes_real_case():
    # GIVEN
    mn = pn.create_small_test_multinet()
    ppmc.run_control_multinet.run_control(mn, max_iter=30, mode="all")
    model_chp = CHPNode(0, mn)
    model_p2g = P2GNode(1, mn)
    model_heat = HeatExchangerNode(0, mn["nets"]["heat"])

    agent_chp = CellAgentRole(model_chp, PowerGasHeatCA(model_chp))
    agent_p2g = CellAgentRole(model_p2g, PowerGasCA(model_p2g))
    agent_heat = CellAgentRole(model_heat, HeatCA(model_heat))

    topology = nx.Graph()
    topology.add_node("chp", agent=agent_chp)
    topology.add_node("p2g", agent=agent_p2g)
    topology.add_node("heat", agent=agent_heat)
    topology.add_edge("chp", "p2g")
    topology.add_edge("heat", "p2g")
    router = SecmesAgentRouter(topology)
    region_m = SecmesRegionManager()
    big_region = region_m.register_region([], "p2g")
    region_m.register_agent("heat", big_region)
    region_m.register_region([], "chp")

    agent_chp.bind(RoleAgentContext(None, RoleHandler(None, None), "chp", None, None))
    agent_p2g.bind(RoleAgentContext(None, RoleHandler(None, None), "p2g", None, None))
    agent_heat.bind(RoleAgentContext(None, RoleHandler(None, None), "heat", None, None))
    agent_chp.secmes_setup(region_manager=region_m, sync_router=router)
    agent_p2g.secmes_setup(region_manager=region_m, sync_router=router)
    agent_heat.secmes_setup(region_manager=region_m, sync_router=router)
    agent_chp.setup()
    agent_p2g.setup()
    agent_heat.setup()

    # WHEN
    attraction = agent_chp.calc_agent_attraction("p2g", np.array([10, 10, -10]))

    # THEN
    assert (attraction > 0).all()
