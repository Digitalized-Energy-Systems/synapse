from peext.node import RegulatableController

from pandapipes.properties.fluids import get_fluid


class EdgeCollector:
    """Collector for edge data regarding a specific controller node"""

    def __init__(self, time_delta=0):
        self.__time_delta = time_delta

    def collect(self, controller: RegulatableController):
        """Collect edge data for the controller

        :param controller: the controller
        :type controller: RegulatableController
        :return: dictionary containing the relevant edge data
        :rtype: Dict
        """
        value_dict = {}
        if "power" in controller.edges:
            edges_power = controller.edges["power"][0]
            if edges_power[1] == "to":
                value_dict["voltage"] = edges_power[0].voltage_magnitude_end(
                    time_delta=self.__time_delta
                )
            else:
                value_dict["voltage"] = edges_power[0].voltage_magnitude_start(
                    time_delta=self.__time_delta
                )
        if "gas" in controller.edges:
            edges_gas = controller.edges["gas"][0]
            if edges_gas[1] == "to":
                value_dict["pressure"] = edges_gas[0].pressure_end(
                    time_delta=self.__time_delta
                )
            else:
                value_dict["pressure"] = edges_gas[0].pressure_start(
                    time_delta=self.__time_delta
                )
        if "heat" in controller.edges:
            edges_heat = controller.edges["heat"][0]
            if edges_heat[1] == "to":
                value_dict["temp"] = edges_heat[0].temp_end(
                    time_delta=self.__time_delta
                )
            else:
                value_dict["temp"] = edges_heat[0].temp_start(
                    time_delta=self.__time_delta
                )
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


def conversion_factor_kgps_to_mw(net):
    fcv = get_fluid(net).get_property("hhv")
    return fcv * 3600 / 1e3
