import peext.scenario.network as ps
import pandapipes

import pandapipes.multinet.control as ppmc
import synapse.simulation.profiles as ssp


def test_multinet_pickle_json():
    feature_rich_mn = ps.create_medium_multinet_gas_power()

    pandapipes.to_json(feature_rich_mn, "network.p")


def test_load_net():
    mn = None
    # mn = pandapipes.from_pickle("data/dat/2022-09-09+14-29-42.134264/1-MV-urban--1-no_sw_1_1/AdaptionRateExperiment-MES/Param-0.5-SplittingStrategy.DISINTEGRATE/network.p")
    print(mn)


def test_tt():

    grid_code = "1-LV-rural1--0-no_sw"

    mn = ps.generate_multi_network_based_on_simbench(
        grid_code,
        heat_deployment_rate=1,
        gas_deployment_rate=0.5,
        chp_density=0.7,
        p2g_density=0.5,
        p2h_density=0.7,
    )
    import pandapipes as pp

    ppmc.run_control_multinet.run_control(mn, max_iter=30, mode="all")

    print("asda")
