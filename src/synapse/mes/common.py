"""Small evaluation / messaging helpers, monee-based.

The former pandapipes/peext couplings are gone: edge state is now read from
monee branch result models (plain ``model.values`` dicts) and the gas energy
conversion uses monee's lgas HHV constant.
"""

# Default lgas HHV (matches monee.network.mes): 15.3 kWh/kg * 3.6 MJ/kWh.
GAS_HHV_MJ_PER_KG = 15.3 * 3.6


class EdgeCollector:
    """Collect the boundary physical quantity (voltage / pressure / temperature)
    a coupling point sees on each of the grids it connects.

    ``branch_values_by_grid`` maps a grid key (``"power"`` / ``"gas"`` / ``"heat"``)
    to the monee branch ``model.values`` dict adjacent to the coupling point.
    """

    def collect(self, branch_values_by_grid):
        value_dict = {}
        power = branch_values_by_grid.get("power")
        if power is not None:
            value_dict["voltage"] = power.get("vm_pu")
        gas = branch_values_by_grid.get("gas")
        if gas is not None:
            value_dict["pressure"] = gas.get("pressure_pu", gas.get("p_pu"))
        heat = branch_values_by_grid.get("heat")
        if heat is not None:
            value_dict["temp"] = heat.get("t_k")
        return value_dict


class MESEvaluator:
    """Evaluates a rule."""

    def __init__(self, ideal_list) -> None:
        self._ideal_list = ideal_list

    def eval(self, obs):
        fitness = 0
        for ideal_tuple in self._ideal_list:
            fitness -= abs(obs[ideal_tuple[1]] - ideal_tuple[0])
        return fitness


class NodeConfigurationMessage:
    def __init__(self, configuration) -> None:
        self._configuration = configuration

    @property
    def configuration(self):
        return self._configuration


def conversion_factor_kgps_to_mw():
    """kg/s -> MW for lgas via the higher heating value."""
    return GAS_HHV_MJ_PER_KG
