"""Short-period run of the adaption experiment.

Identical pipeline to :mod:`adaption` -- centrality-weighted simbench MES build,
DAT coalition-formation simulation, per-run persistence of the monee network and
region history -- but with a small ``TIME_STEPS`` so it finishes in a few
minutes.  Results are written under the *synapse repo root* using the same
``data/dat/...`` layout as ``adaption.py`` (rather than relative to the cwd).

Run from anywhere with the synapse sources on the path, e.g.::

    PYTHONPATH=src python experiments/cs/run_short_adaption.py
"""

import random
from datetime import datetime
from pathlib import Path

import numpy as np

from synapse.agent.dat import SplittingStrategy
from synapse.simulation.scenarios import start_dat_simulation

# adaption.py lives next to this script; reuse its MES builder and result writer.
from adaption import create_common_mes, write_results

# Repo root = .../synapse (this file is experiments/cs/run_short_adaption.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATE_TIME_STR = str(datetime.now()).replace(" ", "+").replace(":", "-")
EXPERIMENT_NAME_MES = "data/dat/{}/{}/AdaptionRateExperiment-MES/Param"

SEED = 100
TIME_STEPS = 3  # short period (adaption.py uses 96)
ADAPTION_RATES = [0.5]
STRATEGIES = [
    SplittingStrategy.DISINTEGRATE,
    SplittingStrategy.CONNECTED_COMPONENTS,
]


def simulate_adaption_rates(simbench_id, cp_density_coeff, deployment_rate_coeff):
    rel = EXPERIMENT_NAME_MES.format(
        DATE_TIME_STR, f"{simbench_id}_{cp_density_coeff}_{deployment_rate_coeff}"
    )
    mes_path = REPO_ROOT / rel

    for strategy in STRATEGIES:
        for adaption_rate in ADAPTION_RATES:
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
            out = f"{mes_path}-{adaption_rate}-{strategy}"
            write_results(out, behavior)
            print(
                f"[done] rate={adaption_rate} strategy={strategy.name} "
                f"regions={behavior.region_manager.region_count} -> {out}"
            )


if __name__ == "__main__":
    simulate_adaption_rates("1-MV-urban--1-no_sw", 1, 1)
    print(f"\nAll runs written under: {REPO_ROOT / 'data' / 'dat' / DATE_TIME_STR}")
