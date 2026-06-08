"""Adaption-rate experiment on the mango + monee stack.

Builds a centrality-weighted multi-energy network from a simbench grid and runs
the DAT coalition-formation scenario across a range of coupling-point change
probabilities and splitting strategies, persisting the monee network and the
resulting region history per run.
"""

import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from monee.io.native import write_omef_network

from synapse.agent.dat import SplittingStrategy
from synapse.mes.network import create_mes_from_simbench_with_cp_distribution
from synapse.simulation.scenarios import start_dat_simulation

ADAPTION_RATES_TO_TEST = np.arange(0, 1, 0.01)
STRATEGIES_TO_TEST = [
    SplittingStrategy.DISINTEGRATE,
    SplittingStrategy.CONNECTED_COMPONENTS,
]
DATE_TIME_STR = str(datetime.now()).replace(" ", "+").replace(":", "-")
EXPERIMENT_NAME_MES = "data/dat/{}/{}/AdaptionRateExperiment-MES/Param"
SEED = 100
TIME_STEPS = 96


def create_common_mes(simbench_id, cp_density_coeff, deployment_rate_coeff):
    return create_mes_from_simbench_with_cp_distribution(
        simbench_id,
        heat_deployment_rate=0.5 * deployment_rate_coeff,
        gas_deployment_rate=0.4 * deployment_rate_coeff,
        chp_density=0.6 * cp_density_coeff,
        p2g_density=0.5 * cp_density_coeff,
        p2h_density=0.3 * cp_density_coeff,
        seed=SEED,
    )


def write_results(path_str, behavior):
    Path(path_str).mkdir(exist_ok=True, parents=True)
    write_omef_network(path_str + "/mes_network.json", behavior.net)
    with open(path_str + "/region-result.p", "wb") as f:
        pickle.dump(behavior.region_manager.get_data_as_copy(), f)


def simulate_adaption_rates(
    simbench_id,
    cp_density_coeff,
    deployment_rate_coeff,
    adaption_rates_override=None,
):
    mes_path_str = EXPERIMENT_NAME_MES.format(
        DATE_TIME_STR, f"{simbench_id}_{cp_density_coeff}_{deployment_rate_coeff}"
    )

    rates = (
        adaption_rates_override
        if adaption_rates_override is not None
        else ADAPTION_RATES_TO_TEST
    )
    for strategy in STRATEGIES_TO_TEST:
        for adaption_rate in rates:
            np.random.seed(SEED)
            random.seed(SEED)
            net = create_common_mes(
                simbench_id, cp_density_coeff, deployment_rate_coeff
            )
            _world, behavior = start_dat_simulation(
                net,
                cp_change_prob=adaption_rate,
                splitting_strategy=strategy,
                time_steps=TIME_STEPS,
            )
            write_results(f"{mes_path_str}-{adaption_rate}-{strategy}", behavior)


if __name__ == "__main__":
    num = None
    date = None
    if len(sys.argv) > 2:
        num = int(sys.argv[1])
        date = str(sys.argv[2])

    if num is not None and date is not None:
        DATE_TIME_STR = date
        simulate_adaption_rates(
            "1-MV-urban--1-no_sw", 1, 1, adaption_rates_override=[num / 100]
        )
    else:
        simulate_adaption_rates(
            "1-MV-urban--1-no_sw", 1, 1, adaption_rates_override=[0.5]
        )
