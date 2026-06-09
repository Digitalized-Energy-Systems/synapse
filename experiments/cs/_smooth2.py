"""Replicate monee's working smooth-gekko recipe on synapse's network.

Matches tests/solver/test_gekko_ipopt_smooth.py::test_smooth_simbench_mes_solves_under_ipopt:
  - node_based_heat_loads=True + HG coupling variants (synapse already does this)
  - apply smooth Weymouth (gas) + smooth Darcy-Weisbach (heat); DO NOT apply MISOCP
  - solve GEKKOSolver(solver=IPOPT).solve(net, exclude_unconnected_nodes=True)
"""

import random
import sys
import time

import numpy as np

from monee.model.formulation import (
    make_smooth_darcy_weisbach_network_formulation,
    make_smooth_weymouth_network_formulation,
)
from monee.solver import GEKKOSolver

import synapse.mes.network as sn
from adaption import create_common_mes

IPOPT = 3
_pos = [a for a in sys.argv[1:] if not a.startswith("--")]
FRICTION = _pos[0] if _pos else "constant"
GRID = _pos[1] if len(_pos) > 1 else "1-MV-rural--0-no_sw"


def smooth_prepare(net):
    sn._clamp_zero_length_pipes(net)
    net.apply_formulation(make_smooth_weymouth_network_formulation(friction_model=FRICTION))
    net.apply_formulation(make_smooth_darcy_weisbach_network_formulation(friction_model=FRICTION))
    return net


sn.prepare_for_solve = smooth_prepare


def main():
    np.random.seed(100)
    random.seed(100)
    print(f"grid={GRID} friction={FRICTION}", flush=True)
    net = create_common_mes(GRID, 1, 1)
    print(
        f"  nodes={len(net.nodes)} childs={len(net.childs)} branches={len(net.branches)}",
        flush=True,
    )
    t = time.perf_counter()
    try:
        res = GEKKOSolver(solver=IPOPT).solve(net, exclude_unconnected_nodes=True)
        dt = time.perf_counter() - t
        print(f"\nSOLVED in {dt:.1f}s  success={getattr(res, 'success', '?')}", flush=True)
        wp = res.dataframes.get("WaterPipe")
        if wp is not None and "t_from_pu" in wp.columns:
            print(
                f"  WaterPipe mass_flow: min={wp['mass_flow'].min():.4g} max={wp['mass_flow'].max():.4g}",
                flush=True,
            )
        jn = res.dataframes.get("Junction")
        if jn is not None and "t_pu" in jn.columns:
            print(f"  Junction t_pu: min={jn['t_pu'].min():.4g} max={jn['t_pu'].max():.4g}", flush=True)
    except Exception as e:  # noqa: BLE001
        dt = time.perf_counter() - t
        print(f"\nFAILED after {dt:.1f}s: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
