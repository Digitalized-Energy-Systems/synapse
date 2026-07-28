# SYNAPSE

**SY**nergistic i**N**fluence of **A**daptive coupling **P**oints in multi-energy **S**ystem int**E**gration

An agent-based simulation of **self-organising cells (coalitions) in multi-energy systems**. Every
component of a coupled electricity/heat/gas grid gets an agent; agents continuously re-negotiate
which coalition ("region") they belong to based on the energy balance of their neighbourhood. On top
of that, coupling points (CHP, P2G, G2P, P2H) switch themselves on and off over time, which tears
holes into both the physical and the agent topology — the research question is how this *adaption
rate* and the chosen *splitting strategy* affect the emerging coalition structure.

Physics comes from [pandapower]/[pandapipes] via [peext] (multi-energy network abstraction), the
multi-agent layer is [mango].

---

## Install

The dependency pins in [pyproject.toml](pyproject.toml) are strict and old
(`pandapower==3.2.1`, `pandapipes==0.12.0`, `mango-agents==0.3.0`, `plotly==5.11.0`,
`kaleido==0.2.1`). **Install into a dedicated virtual environment** — mango 0.3 uses the pre-1.0 API
(`mango.core.container`) and will break any environment that holds a newer mango, pandapower or
plotly.

Python 3.11 is the interpreter this code is developed against; the pin chain
(`mango-agents 0.3.0` → `protobuf 3.13.0`) resolves there, newer interpreters are untested.

```powershell
# Windows / PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Extras:

- `pip install -e ".[dev]"` — pytest + twine.
- `pip install monee` — required only by [src/synapse/evaluation/evaluation.py](src/synapse/evaluation/evaluation.py),
  which imports `monee` for the network plots but does not declare it as a dependency.

## Run a simulation

Smallest thing that actually simulates something — a small SimBench LV grid, extended into a
multi-energy network, with coupling-point agents that toggle with probability 0.5 per step:

```python
import peext.scenario.network as ps

from synapse.agent.dat import SplittingStrategy
from synapse.simulation.scenarios import start_dat_simulation

multinet = ps.generate_multi_network_based_on_simbench(
    "1-LV-rural3--1-no_sw",
    heat_deployment_rate=0.5,
    gas_deployment_rate=0.4,
    chp_density=0.6,
    p2g_density=0.1,
    p2h_density=0.3,
)

start_dat_simulation(
    multinet,
    time_steps=10,
    cp_change_prob=0.5,
    splitting_strategy=SplittingStrategy.CONNECTED_COMPONENTS,
    name="out/quickstart",
)
```

Save as `run_quickstart.py` and `python run_quickstart.py`. Results land in `out/quickstart/`.

The full experiment — a sweep over adaption rates × splitting strategies on the MV urban grid — is
[experiments/cs/adaption.py](experiments/cs/adaption.py):

```bash
python experiments/cs/adaption.py            # single config, adaption rate 0.5, 96 steps
python experiments/cs/adaption.py 25 run-a   # rate 0.25 across four CP-density/deployment configs,
                                             # results grouped under the run tag "run-a"
```

The second form exists to shard the sweep across processes: the CLI argument doubles as the mango
container port offset (`port_add`), so several runs can execute in parallel without port clashes.

### Knobs

| Parameter | Where | Meaning |
|---|---|---|
| `cp_change_prob` | [scenarios.py](src/synapse/simulation/scenarios.py) | Per-step probability that a coupling point flips on/off — the *adaption rate*. |
| `splitting_strategy` | [agent/dat.py](src/synapse/agent/dat.py) | `DISINTEGRATE` or `CONNECTED_COMPONENTS`, see below. |
| `time_steps` | [scenarios.py](src/synapse/simulation/scenarios.py) | Simulation length (default 96 = one day at 15 min). Agents only act from step 2, coupling points from step 3. |
| `demand_attacher_function` | [scenarios.py](src/synapse/simulation/scenarios.py) | Defaults to random profiles; pass `None` and attach SimBench/measured profiles yourself ([profiles.py](src/synapse/simulation/profiles.py)). |
| `no_energy_flow` | [scenarios.py](src/synapse/simulation/scenarios.py) | Skip the multinet power/pipe flow entirely — fast runs of the pure coalition dynamics. |
| `port_add` | [scenarios.py](src/synapse/simulation/scenarios.py) | Offset on the mango container port `127.0.0.2:5555`. |
| `neighborhood_size`, `cutoff_length` | [agent/core.py](src/synapse/agent/core.py) | How far an agent looks for coalition partners: at most 10 agents, Dijkstra cutoff 0.1 on the loss-weighted agent graph. |

### Output

Each run writes into its `name` directory:

- `agent-result.p` — pickle of `(region_history, agent_state_data, topology_graph_history)`:
  the per-step region graph (region id → assigned agents), per-agent time series
  (`region_balance_power/heat/gas`, `attraction_*` per neighbour, current `region`), and a per-step
  copy of the agent topology.
- `network-result.p` — per-step physical measurements collected by peext's `StaticPlottingController`
  (omitted when `no_energy_flow=True`).

[src/synapse/evaluation/evaluation.py](src/synapse/evaluation/evaluation.py) provides the plotly
figure builders and `write_all_in_one(...)`, which renders a set of figures into a single
`report.html` plus one PDF per figure. Note it writes a throwaway `random_figure.pdf` into the
working directory as a kaleido warm-up.

`data/` and `out/` are gitignored — they hold local run artefacts only.

## How it works

**Agent topology.** [scenario/cell_agents.py](src/synapse/simulation/scenario/cell_agents.py) turns
the peext `MENetwork` into a `networkx.MultiGraph`: one node per component, id
`network:component_type:id`. Coupling points are a special case — a single agent, but one graph node
per carrier it touches (`<aid>#power`, `<aid>#gas`, …) connected by edges weighted with the
conversion efficiency. Buses/junctions stay in the graph as *virtual* nodes for routing but get no
agent. Every step, edge weights are refreshed with the actual physical losses, so "communication
distance" tracks the state of the grid.

**Coalition formation** ([agent/cell_agent.py](src/synapse/agent/cell_agent.py)). Each agent knows
its own balance as a 3-vector (power, heat, gas). Per step it sums the balance of its coalition,
then for every neighbour computes an *attraction* — essentially `-sign(own balance) · other balance`,
so surplus attracts deficit and equal signs repel. If a neighbour is attractive in all three
carriers, it receives a `JoinRequest`; the receiver switches coalition when the requesting region is
at least as attractive as its current one. Agents that end up unassigned open a fresh region.
`SecmesRegionManager` ([agent/core.py](src/synapse/agent/core.py)) is the shared bookkeeping of
region membership and adjacency.

**Adaptive coupling points** ([agent/dat.py](src/synapse/agent/dat.py)). A `DATCouplingPointRole`
flips its component with probability `cp_change_prob`. Switching *on* means `regulate(1)`, relinking
its edges and opening a new region. Switching *off* means `regulate(0)`, unlinking its edges from the
agent graph, and then one of:

- `DISINTEGRATE` — the coupling point leaves its region; if the rest is still connected it simply
  drops out, otherwise the region is dissolved and every member is notified
  (`RegionDisbandedMessage`) so it can re-plan in the same step.
- `CONNECTED_COMPONENTS` — the region is always dissolved and immediately re-created as one region
  per remaining connected component.

**Main loop** ([agent/world.py](src/synapse/agent/world.py)). `AsyncWorld` steps profile
controllers, runs the multinet flow calculation, then triggers the plotting, agent and fault
controllers — agent decisions therefore always see the previous step's physical state. Message
passing is synchronous and direct (`SecmesAgentRouter.dispatch_message_sync`) while the mango
container provides agent identity and lifecycle.

## Repository layout

| Path | Contents |
|---|---|
| [src/synapse/agent/core.py](src/synapse/agent/core.py) | Region manager, agent router/topology, pandapower controller bridge (`AgentController`). |
| [src/synapse/agent/cell_agent.py](src/synapse/agent/cell_agent.py) | `CellAgentRole` (attraction, join protocol) and per-carrier balance accessors. |
| [src/synapse/agent/dat.py](src/synapse/agent/dat.py) | Dynamic adaptive topology: splitting strategies and the coupling-point role. |
| [src/synapse/agent/world.py](src/synapse/agent/world.py) | `AsyncWorld` / `SyncWorld` simulation loops and result writing. |
| [src/synapse/simulation/](src/synapse/simulation/) | Scenario entry points, agent-topology construction, demand profiles, fault injection. |
| [src/synapse/mes/](src/synapse/mes/) | Multi-energy network generation with centrality-driven coupling-point placement. |
| [src/synapse/cn/network.py](src/synapse/cn/network.py) | Complex-network views (physical bus/junction graph, plots) of an MES. |
| [src/synapse/da/dgd.py](src/synapse/da/dgd.py) | Distributed gradient descent used for the (currently experimental) operation-point negotiation. |
| [src/synapse/evaluation/evaluation.py](src/synapse/evaluation/evaluation.py) | Plotly figure builders, network plots, HTML/PDF report writer. |
| [src/synapse/data/](src/synapse/data/) | SQLite result store and a trivial step observer. |
| [experiments/cs/](experiments/cs/) | Runnable experiment scripts. |
| [test/](test/) | pytest suite. |

## Tests

```bash
pytest test
```

[test/agent/cell_agent_test.py](test/agent/cell_agent_test.py) pins down the attraction semantics
per carrier and component type, [test/da/dgd_test.py](test/da/dgd_test.py) covers the gradient
descent, [test/multinet_load_save_test.py](test/multinet_load_save_test.py) exercises multinet
generation and serialisation. Several of these build real networks and run flow calculations, so the
suite is not instant.

## Known rough edges

- [src/synapse/main_dat.py](src/synapse/main_dat.py) imports `synapse.scenario.scenarios`, which no
  longer exists (the module is `synapse.simulation.scenarios`) — use the snippet above or
  [experiments/cs/adaption.py](experiments/cs/adaption.py) instead.
- `evaluation.py` imports `monee` at module level without declaring it in `pyproject.toml`.
- The USA residential heat/gas profiles
  (`data/input/profiles/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/…`) are not part of the repository; when
  missing, [profiles.py](src/synapse/simulation/profiles.py) logs a warning and silently falls back
  to random profiles.
- Fault injection (`Fault`, `FaultExecutor`, `FaultInjector` in
  [scenario/fault.py](src/synapse/simulation/scenario/fault.py)) is implemented but not wired up by
  `start_dat_simulation`; `AsyncWorld` has to be constructed directly to use it.
- The evaluation scripts that produced the reports under `data/` are not in this snapshot — only the
  adaption sweep remains in `experiments/cs/`.

[pandapower]: https://www.pandapower.org/
[pandapipes]: https://www.pandapipes.org/
[peext]: https://pypi.org/project/peext/
[mango]: https://mango-agents.readthedocs.io/
