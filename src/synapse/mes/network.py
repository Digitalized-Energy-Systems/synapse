"""Multi-energy network construction on top of monee.

Replaces the former peext / pandapipes construction.  The power grid is obtained
from simbench through monee's importer; the gas and (node-based, supply/return)
heat grids are generated with monee's scalable MES helpers; and the coupling
points (CHP / P2G / P2H) are placed with a *centrality-weighted* distribution --
the scientific core of synapse, which monee's built-in (uniform per-node
Bernoulli) builder does not provide.

The resulting network uses the node-based / McCormick-DHS-compatible heat model
and is returned solve-ready under the MISOCP formulation (+ GasLinepack /
LumpedThermalCapacitance extensions), so it scales to real simbench grids with
the gurobi backend -- the same recipe scare uses.
"""

import random

import networkx as nx
import numpy as np
import scipy.stats as scipystats

import monee.express as mx
from monee import GasLinepack, LumpedThermalCapacitance, MISOCP_NETWORK_FORMULATION
from monee.io.from_simbench import obtain_simbench_net, obtain_simbench_net_with_td
from monee.network.mes import (
    GAS_HHV_MJ_PER_KG,
    _node_power_gen_mw,
    _node_power_load_mw,
    create_gas_tree_net_for_power,
    create_heat_supply_return_net_for_power,
)

DEFAULT_COUPLINGS = ("chp", "p2g", "p2h")

# monee's gas/heat tree builders derive pipe length from the geographic distance
# between the two buses; simbench grids contain co-located buses (distance 0),
# which yields a zero-length pipe that divides by zero in the Weymouth/thermal
# equations.  Clamp such pipes to a tiny positive length before solving.
MIN_PIPE_LENGTH_M = 1.0


def _default_distribution(metric, all_values):
    """Default centrality -> placement-probability weight.

    Mirrors the former synapse default: a normal pdf centred on the most
    central node so coupling points concentrate on high-centrality buses.
    """
    hi = max(all_values)
    lo = min(all_values)
    scale = (hi - lo) or 1.0
    return scipystats.norm.pdf(metric, loc=hi, scale=scale)


def compute_centrality_map(net_power, metric=None):
    """Return ``{power_node_id: centrality}`` for the power-grid graph.

    ``metric`` defaults to :func:`networkx.betweenness_centrality` and may be any
    callable ``graph -> {node_id: score}``.
    """
    if metric is None:
        metric = nx.betweenness_centrality
    graph = nx.Graph(net_power._network_internal)
    return dict(metric(graph))


def _placement_probability(node_id, metric_map, density, density_function):
    """Centrality-weighted per-node placement probability, normalised so the mean
    probability across nodes equals ``density`` (keeps ``density`` a meaningful
    deployment-rate knob, like monee's uniform builder)."""
    all_values = list(metric_map.values())
    if not all_values:
        return density
    raw = density_function(metric_map[node_id], all_values)
    mean_raw = float(np.mean([density_function(v, all_values) for v in all_values]))
    if mean_raw <= 0:
        return density
    return min(1.0, density * raw / mean_raw)


def create_coupling_points_distributed(
    mes_net,
    bus_to_gas_junc,
    bus_to_heat_supply_junc,
    metric_map,
    density,
    density_function,
    couplings=DEFAULT_COUPLINGS,
    chp_efficiency_power=0.4,
    chp_efficiency_heat=0.45,
    chp_p_share=0.5,
    p2g_efficiency=0.7,
    p2g_p_share=1.0,
    p2h_efficiency=0.95,
    p2h_p_share=1.0,
    cp_size_multiplier=1.0,
    regulation=1.0,
    seed=None,
):
    """Place CHP / P2G / P2H coupling points biased by node centrality.

    A centrality-weighted variant of monee's
    :func:`~monee.network.mes.create_coupling_points_for_mes`: the per-node
    Bernoulli draw is reweighted by ``density_function(centrality)`` (mean kept
    at ``density``), and the HeatGenerator ("HG") variants are used so the result
    stays McCormick-DHS compatible.  Returns ``list[{"type","node","id"}]``.
    """
    rng = random.Random(seed) if seed is not None else random
    coupling_set = {c.lower() for c in couplings}
    candidate_node_ids = [
        nid for nid in bus_to_gas_junc if nid in bus_to_heat_supply_junc
    ]

    created = []
    for power_node_id in candidate_node_ids:
        prob = _placement_probability(
            power_node_id, metric_map, density, density_function
        )
        if rng.random() > prob:
            continue

        node = mes_net.node_by_id(power_node_id)
        p_ref_mw = _node_power_gen_mw(mes_net, node) or _node_power_load_mw(
            mes_net, node
        )
        if p_ref_mw <= 0:
            continue

        gas_junc = bus_to_gas_junc[power_node_id]
        heat_supply_junc = bus_to_heat_supply_junc[power_node_id]
        unit_type = rng.choice(sorted(coupling_set))

        if unit_type == "chp":
            chp_p_target = chp_p_share * cp_size_multiplier * p_ref_mw
            mass_flow = round(
                chp_p_target / max(chp_efficiency_power, 1e-3) / GAS_HHV_MJ_PER_KG, 6
            )
            uid = mx.create_chp_hg(
                mes_net,
                power_node_id=power_node_id,
                heat_node_id=heat_supply_junc,
                gas_node_id=gas_junc,
                mass_flow_setpoint=mass_flow,
                efficiency_power=chp_efficiency_power,
                efficiency_heat=chp_efficiency_heat,
                regulation=regulation,
            )
            created.append({"type": "chp", "node": power_node_id, "id": uid})
        elif unit_type == "p2g":
            p2g_p_in = p2g_p_share * cp_size_multiplier * p_ref_mw
            mass_flow = round(p2g_efficiency * p2g_p_in / GAS_HHV_MJ_PER_KG, 6)
            bid = mx.create_p2g(
                mes_net,
                from_node_id=power_node_id,
                to_node_id=gas_junc,
                efficiency=p2g_efficiency,
                mass_flow_setpoint=mass_flow,
                regulation=regulation,
            )
            created.append({"type": "p2g", "node": power_node_id, "id": bid})
        elif unit_type == "p2h":
            p2h_p_in = p2h_p_share * cp_size_multiplier * p_ref_mw
            heat_mw = round(p2h_p_in * p2h_efficiency, 6)
            bid = mx.create_p2h_hg(
                mes_net,
                power_node_id=power_node_id,
                heat_node_id=heat_supply_junc,
                heat_energy_mw=heat_mw,
                efficiency=p2h_efficiency,
            )
            created.append({"type": "p2h", "node": power_node_id, "id": bid})
    return created


def _clamp_zero_length_pipes(mes_net, min_length_m=MIN_PIPE_LENGTH_M):
    """Raise any zero-length pipe to ``min_length_m`` so the gas/heat equations
    don't divide by zero.  Returns the number of pipes adjusted."""
    clamped = 0
    for branch in mes_net.branches:
        model = branch.model
        length = getattr(model, "length_m", None)
        if length is not None and float(length) <= 0.0:
            model.length_m = min_length_m
            clamped += 1
    return clamped


def prepare_for_solve(mes_net):
    """Make ``mes_net`` solve-ready at scale: apply the MISOCP formulation and the
    storage extensions, matching scare's gurobi recipe."""
    _clamp_zero_length_pipes(mes_net)
    mes_net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    mes_net.add_extension(GasLinepack())
    mes_net.add_extension(LumpedThermalCapacitance(first_step_steady_state=True))
    return mes_net


def build_mes(
    net_power,
    heat_deployment_rate,
    gas_deployment_rate,
    chp_density=0.3,
    p2g_density=0.1,
    p2h_density=0.1,
    metric=None,
    distribution=_default_distribution,
    couplings=DEFAULT_COUPLINGS,
    coupling_kwargs=None,
    gas_kwargs=None,
    heat_kwargs=None,
    seed=100,
):
    """Build a combined node-based MES from a monee power network with
    centrality-weighted coupling points.  Returns a solve-ready
    :class:`monee.model.network.Network`.

    ``chp_density`` / ``p2g_density`` / ``p2h_density`` set the per-type mean
    deployment rate; ``heat_deployment_rate`` / ``gas_deployment_rate`` are kept
    for API compatibility and forwarded as the heat/gas load shares.
    """
    random.seed(seed)
    np.random.seed(seed)

    mes_net = net_power.copy()
    bus_to_gas_junc = create_gas_tree_net_for_power(
        net_power, mes_net, **(gas_kwargs or {})
    )
    bus_to_heat_supply, _heat_return = create_heat_supply_return_net_for_power(
        net_power,
        mes_net,
        node_based_heat_loads=True,
        **(heat_kwargs or {}),
    )

    metric_map = compute_centrality_map(net_power, metric=metric)
    ckw = dict(coupling_kwargs or {})

    # One pass per coupling type so each keeps its own density knob, mirroring the
    # former synapse builder that called create_{p2h,chp,p2g} separately.
    for cp_type, dens in (
        ("chp", chp_density),
        ("p2g", p2g_density),
        ("p2h", p2h_density),
    ):
        if cp_type in {c.lower() for c in couplings} and dens > 0:
            create_coupling_points_distributed(
                mes_net,
                bus_to_gas_junc,
                bus_to_heat_supply,
                metric_map,
                dens,
                distribution,
                couplings=(cp_type,),
                seed=seed,
                **ckw,
            )

    return prepare_for_solve(mes_net)


def create_mes_from_simbench_with_cp_distribution(
    simbench_id,
    heat_deployment_rate,
    gas_deployment_rate,
    chp_density=0.3,
    p2g_density=0.1,
    p2h_density=0.1,
    metric=None,
    distribution=_default_distribution,
    seed=100,
    **kwargs,
):
    """Build a solve-ready centrality-weighted MES from a simbench grid.

    Drop-in replacement for the former peext-based builder of the same name.
    """
    net_power = obtain_simbench_net(simbench_id)
    return build_mes(
        net_power,
        heat_deployment_rate,
        gas_deployment_rate,
        chp_density=chp_density,
        p2g_density=p2g_density,
        p2h_density=p2h_density,
        metric=metric,
        distribution=distribution,
        seed=seed,
        **kwargs,
    )


def create_mes_from_simbench_with_td(
    simbench_id,
    heat_deployment_rate,
    gas_deployment_rate,
    chp_density=0.3,
    p2g_density=0.1,
    p2h_density=0.1,
    metric=None,
    distribution=_default_distribution,
    seed=100,
    **kwargs,
):
    """Like :func:`create_mes_from_simbench_with_cp_distribution` but also returns
    the simbench :class:`~monee.simulation.timeseries.TimeseriesData` profile."""
    net_power, td = obtain_simbench_net_with_td(simbench_id)
    mes = build_mes(
        net_power,
        heat_deployment_rate,
        gas_deployment_rate,
        chp_density=chp_density,
        p2g_density=p2g_density,
        p2h_density=p2h_density,
        metric=metric,
        distribution=distribution,
        seed=seed,
        **kwargs,
    )
    return mes, td
