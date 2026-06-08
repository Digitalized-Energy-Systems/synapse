from synapse.agent.dat import SplittingStrategy
from synapse.mes.network import create_mes_from_simbench_with_cp_distribution
from synapse.simulation.scenarios import start_dat_simulation

if __name__ == "__main__":
    net = create_mes_from_simbench_with_cp_distribution(
        "1-MV-urban--1-no_sw",
        heat_deployment_rate=0.5,
        gas_deployment_rate=0.4,
        chp_density=0.6,
        p2g_density=0.1,
        p2h_density=0.3,
    )
    start_dat_simulation(
        net,
        time_steps=10,
        splitting_strategy=SplittingStrategy.CONNECTED_COMPONENTS,
    )
