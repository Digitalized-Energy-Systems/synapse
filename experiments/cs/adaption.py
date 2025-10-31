from pathlib import Path
import peext.scenario.network as ps
import numpy as np
import random, sys
import simbench
import pandapipes
import pandapipes.multinet as ppm
from datetime import datetime

from synapse.simulation.scenarios import start_dat_simulation
import synapse.simulation.profiles as ssp

from synapse.agent.dat import (
    SplittingStrategy,
)


ADAPTION_RATES_TO_TEST = np.arange(0, 1, 0.01)
STRATEGIES_TO_TEST = [
    SplittingStrategy.DISINTEGRATE,
    SplittingStrategy.CONNECTED_COMPONENTS,
]
DATE_TIME_STR = str(datetime.now()).replace(" ", "+").replace(":", "-")
EXPERIMENT_NAME_MES = "data/dat/{}/{}/AdaptionRateExperiment-MES/Param"
EXPERIMENT_NAME_BASE = "data/dat/{}/{}/AdaptionRateExperiment-BASELINE/Param"
SEED = 100
TIME_STEPS = 96
NO_ENERGY_FLOW = True


def create_common_mn(simbench_id, cp_density_coeff, deployment_rate_coeff):
    return ps.generate_multi_network_based_on_simbench(
        simbench_id,
        heat_deployment_rate=0.5 * deployment_rate_coeff,
        gas_deployment_rate=0.4 * deployment_rate_coeff,
        chp_density=0.6 * cp_density_coeff,
        p2g_density=0.5 * cp_density_coeff,
        p2h_density=0.3 * cp_density_coeff,
    )


def create_power_only_mn(simbench_id):
    power_net = simbench.get_simbench_net(simbench_id)
    power_net.name = "power"

    mn = ppm.create_empty_multinet()
    ppm.add_net_to_multinet(mn, power_net, net_name="power")
    return mn


def create_random_demand(simbench_id, cp_density_coeff, deployment_rate_coeff):
    np.random.seed(SEED)
    random.seed(SEED)
    mn = create_common_mn(simbench_id, cp_density_coeff, deployment_rate_coeff)
    load_mat = ssp.create_random_profile(
        len(mn["nets"]["power"].load.index), TIME_STEPS, only_positive=True
    )
    he_mat = ssp.create_random_profile(
        len(mn["nets"]["heat"].heat_exchanger.index), TIME_STEPS, only_positive=False
    )
    sink_mat = ssp.create_random_profile(
        len(mn["nets"]["gas"].sink.index), TIME_STEPS, only_positive=True
    )
    return load_mat, he_mat, sink_mat


def write_mes_network(
    mes_path_str, simbench_id, cp_density_coeff, deployment_rate_coeff
):
    np.random.seed(SEED)
    random.seed(SEED)
    Path(mes_path_str).mkdir(exist_ok=True, parents=True)
    ser_mn = create_common_mn(simbench_id, cp_density_coeff, deployment_rate_coeff)
    pandapipes.to_pickle(ser_mn, mes_path_str + "/mes_network.p")


def simulate_adaption_rates(
    simbench_id,
    cp_density_coeff,
    deployment_rate_coeff,
    with_baseline=False,
    adaption_rates_override=None,
    port_add=0,
):

    if with_baseline:
        np.random.seed(SEED)
        random.seed(SEED)
        mn_pn = create_power_only_mn(simbench_id)
        path_str_baseline = EXPERIMENT_NAME_BASE.format(
            DATE_TIME_STR, f"{simbench_id}_0_baseline"
        )
        Path(path_str_baseline).mkdir(exist_ok=True, parents=True)
        pandapipes.to_pickle(mn_pn, path_str_baseline + "/baseline_network.p")
        ssp.create_and_attach_all_simbench_profiles(mn_pn["nets"]["power"])
        start_dat_simulation(
            mn_pn,
            demand_attacher_function=None,
            time_steps=TIME_STEPS,
            name=f"{path_str_baseline}-0-0",
            port_add=port_add,
        )

    mes_path_str = EXPERIMENT_NAME_MES.format(
        DATE_TIME_STR, f"{simbench_id}_{cp_density_coeff}_{deployment_rate_coeff}"
    )
    write_mes_network(
        mes_path_str,
        simbench_id,
        cp_density_coeff,
        deployment_rate_coeff,
    )

    for strategy in STRATEGIES_TO_TEST:
        for adaption_rate in (
            adaption_rates_override
            if adaption_rates_override is not None
            else ADAPTION_RATES_TO_TEST
        ):
            np.random.seed(SEED)
            random.seed(SEED)
            mn = create_common_mn(simbench_id, cp_density_coeff, deployment_rate_coeff)
            ssp.create_and_attach_usa_gas_profiles(
                mn["nets"]["gas"], time_steps=TIME_STEPS
            )
            ssp.create_and_attach_usa_heat_profiles(
                mn["nets"]["heat"], time_steps=TIME_STEPS
            )
            ssp.create_and_attach_all_simbench_profiles(mn["nets"]["power"])
            start_dat_simulation(
                mn,
                demand_attacher_function=None,
                time_steps=TIME_STEPS,
                cp_change_prob=adaption_rate,
                splitting_strategy=strategy,
                name=f"{mes_path_str}-{adaption_rate}-{strategy}",
                port_add=port_add,
            )


if __name__ == "__main__":
    num = None
    date = None
    if len(sys.argv) > 2:
        num = int(sys.argv[1])
        date = str(sys.argv[2])

    if num is not None and date is not None:
        DATE_TIME_STR = date
        if num == 0:
            simulate_adaption_rates(
                "1-MV-urban--1-no_sw",
                1,
                1,
                with_baseline=True,
                adaption_rates_override=[],
                port_add=num,
            )  # only baseline

        simulate_adaption_rates(
            "1-MV-urban--1-no_sw",
            1,
            1,
            adaption_rates_override=[num / 100],
            port_add=num,
        )
        simulate_adaption_rates(
            "1-MV-urban--1-no_sw",
            1,
            2,
            adaption_rates_override=[num / 100],
            port_add=num,
        )

        simulate_adaption_rates(
            "1-MV-urban--1-no_sw",
            2,
            1,
            adaption_rates_override=[num / 100],
            port_add=num,
        )
        simulate_adaption_rates(
            "1-MV-urban--1-no_sw",
            2,
            2,
            adaption_rates_override=[num / 100],
            port_add=num,
        )

    else:
        simulate_adaption_rates(
            "1-MV-urban--1-no_sw",
            1,
            1,
            with_baseline=False,
            adaption_rates_override=[0.5],
        )
