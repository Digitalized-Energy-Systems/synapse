from functools import reduce
import scipy.stats as sciypystats
import random
import numpy as np

from peext.network import from_panda_multinet
import peext.scenario.network as psn
import pandapipes.multinet as ppm

import synapse.cn.network as cn
import networkx as nx


def extend_to_mn(net_power, heat_deployment_rate, gas_deployment_rate):
    net_power.name = "power"
    net_heat = psn.create_heat_net_for_power(net_power, heat_deployment_rate)
    net_gas = psn.create_gas_net_for_power(net_power, gas_deployment_rate)

    mn = ppm.create_empty_multinet()
    ppm.add_net_to_multinet(mn, net_heat, net_name="heat")
    ppm.add_net_to_multinet(mn, net_power, net_name="power")
    ppm.add_net_to_multinet(mn, net_gas, net_name="gas")
    return net_heat, net_gas, mn


# TODO implement degree distribution for whole network
# should be variation of the method below


def create_mes_from_simbench_with_cp_distribution(
    simbench_id,
    heat_deployment_rate,
    gas_deployment_rate,
    chp_density=0.3,
    p2g_density=0.1,
    p2h_density=0.1,
    metric=None,
    distribution=lambda metric, all_values: sciypystats.norm.pdf(
        metric, loc=max(all_values), scale=max(all_values) - min(all_values)
    ),
    seed=100,
):
    net_power = psn.get_simbench_net(simbench_id)

    random.seed(seed)
    np.random.seed(seed)

    net_heat, net_gas, mn = extend_to_mn(
        net_power, heat_deployment_rate, gas_deployment_rate
    )

    psn.create_p2h_in_combined_generated_network(mn, net_power, net_heat, 1)
    psn.create_chp_in_combined_generated_network(mn, net_power, net_heat, net_gas, 1)
    psn.create_p2g_in_combined_generated_network(mn, net_power, net_gas, 1)

    me_network = from_panda_multinet(mn)

    if metric == None:
        metric = nx.betweenness_centrality

    bus_junc_nx_graph = cn.to_phys_bus_junc_networkx_graph(me_network)
    metric_map = {
        reduce(
            lambda v1, v2: v1 + "-" + v2[0] + ":" + str(v2[1]),
            []
            if "inner_nodes" not in bus_junc_nx_graph.nodes[name]
            else bus_junc_nx_graph.nodes[name]["inner_nodes"].values(),
            name,
        ): val
        for name, val in dict(metric(bus_junc_nx_graph)).items()
    }

    random.seed(seed)
    np.random.seed(seed)

    net_heat, net_gas, mn = extend_to_mn(
        net_power, heat_deployment_rate, gas_deployment_rate
    )
    psn.create_p2h_in_combined_generated_network(
        mn,
        net_power,
        net_heat,
        p2h_density,
        density_function=distribution,
        node_loc_map=metric_map,
    )
    psn.create_chp_in_combined_generated_network(
        mn,
        net_power,
        net_heat,
        net_gas,
        chp_density,
        density_function=distribution,
        node_loc_map=metric_map,
    )
    psn.create_p2g_in_combined_generated_network(
        mn,
        net_power,
        net_gas,
        p2g_density,
        density_function=distribution,
        node_loc_map=metric_map,
    )

    return mn
