"""Synapse mango environment behavior.

A custom :class:`mango.simulation.environment.Behavior` that integrates a monee
multi-energy network into a mango :class:`~mango.simulation.world.SimulationWorld`:

* runs steady-state energy flow (monee, gurobi/MISOCP) on ``initialize`` and,
  rate-limited, on every dirty ``on_step``;
* applies the per-step demand schedule;
* exposes ``observe(aid)`` / ``act(aid, "regulate", factor)`` for the agents;
* hosts the *shared coalition state* -- the region manager and the agent
  topology graph -- so the coalition-formation roles reach it the same way
  scare roles reach shared maps on their behavior.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import monee
from mango.simulation.environment import Behavior, Environment
from mango.util.clock import Clock

logger = logging.getLogger(__name__)


class SynapseEnvironmentBehavior(Behavior):
    def __init__(
        self,
        net,
        *,
        solver: str | None = "gurobi",
        demand_schedule=None,
        region_manager=None,
        agent_graph=None,
        energy_flow_cooldown_s: float = 0.5,
    ) -> None:
        self._net = net
        self._net_results = None
        self._solver = solver
        self._demand_schedule = demand_schedule
        self._dirty = False
        self._energy_flow_cooldown_s = float(energy_flow_cooldown_s)
        self._last_energy_flow_t = float("-inf")
        # ``_step_index`` is the integer simulated second; demand profiles are
        # indexed by it so they advance on the sim clock rather than on every
        # micro discrete-event step.
        self._step_index = 0

        # aid -> (kind, component_id); kind in {"child", "node", "branch"}
        self._component_by_aid: dict[str, tuple[str, Any]] = {}
        self._observers: dict[str, Callable[[], dict]] = {}
        self._actions: dict[str, dict[str, Callable]] = {}

        # Shared coalition state (set by the world builder).
        self.region_manager = region_manager
        self.agent_graph = agent_graph
        # aid -> mango AgentAddress, filled in by the world builder once agents exist.
        self.aid_to_addr: dict[str, Any] = {}
        # CP aid -> list of grid keys it bridges (for cp neighborhood / splitting).
        self.cp_networks: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ solve
    @property
    def net(self):
        return self._net

    @property
    def net_results(self):
        return self._net_results

    @property
    def step_index(self) -> int:
        return self._step_index

    def _solve(self):
        self._net_results = monee.run_energy_flow(self._net, solver=self._solver)
        self._dirty = False

    def mark_dirty(self) -> None:
        self._dirty = True

    def flush_energy_flow(self) -> None:
        self._solve()

    def initialize(self, environment: Environment, clock: Clock) -> None:
        logger.debug("SynapseEnvironmentBehavior: initial energy flow")
        self._solve()
        self._last_energy_flow_t = clock.time

    def on_step(self, environment: Environment, clock: Clock, step_size_s: float) -> None:
        # Advance the demand profile once per simulated second.
        step = int(clock.time)
        if step > self._step_index:
            self._step_index = step
            if self._demand_schedule is not None and self._demand_schedule.apply(
                self._net, step
            ):
                self._dirty = True

        if self._dirty:
            since = clock.time - self._last_energy_flow_t
            if since < self._energy_flow_cooldown_s:
                return
            self._solve()
            self._last_energy_flow_t = clock.time

    # --------------------------------------------------------------- install
    def install(self, agent, **kwargs) -> None:
        component_id = kwargs.get("id")
        component_type = kwargs.get("type")
        if component_type is None:
            return
        aid = agent.aid
        self._component_by_aid[aid] = (component_type, component_id)
        if component_type == "child":
            self._install_child(aid, component_id)
        elif component_type == "node":
            self._install_node(aid, component_id)
        elif component_type == "branch":
            self._install_branch(aid, component_id)

    def _results_net(self):
        return self._net_results.network if self._net_results is not None else self._net

    def _install_child(self, aid, child_id):
        def observer():
            net = self._results_net()
            child = net.child_by_id(child_id)
            node = net.node_by_id(child.node_id)
            return {**dict(node.model.values), **dict(child.model.values)}

        self._observers[aid] = observer

        def regulate(factor):
            self._net.child_by_id(child_id).model.regulation = factor
            self._dirty = True

        self._actions[aid] = {"regulate": regulate}

    def _install_node(self, aid, node_id):
        def observer():
            net = self._results_net()
            return dict(net.node_by_id(node_id).model.values)

        self._observers[aid] = observer
        self._actions[aid] = {}

    def _install_branch(self, aid, branch_id):
        def observer():
            net = self._results_net()
            branch = net.branch_by_id(branch_id)
            return dict(branch.model.values)

        self._observers[aid] = observer

        def regulate(factor):
            self._net.branch_by_id(branch_id).model.regulation = factor
            self._dirty = True

        self._actions[aid] = {"regulate": regulate}

    # --------------------------------------------------------------- observe
    def observe(self, aid: str) -> dict:
        fn = self._observers.get(aid)
        return fn() if fn is not None else {}

    def act(self, aid: str, action: str, *args, **kwargs) -> None:
        fn = self._actions.get(aid, {}).get(action)
        if fn is not None:
            fn(*args, **kwargs)
        else:
            logger.warning("No action %r registered for agent %r", action, aid)

    def has_action(self, aid: str, action: str) -> bool:
        return action in self._actions.get(aid, {})

    def component_of(self, aid: str):
        return self._component_by_aid.get(aid)

    def address_of(self, aid: str):
        return self.aid_to_addr.get(aid)
