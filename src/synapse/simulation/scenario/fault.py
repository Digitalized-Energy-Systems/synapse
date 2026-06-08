"""Fault injection for the monee-based simulation.

The former pandapower ``Controller`` is gone.  A :class:`Fault` is applied by
mutating the monee network (e.g. deactivating a branch / node) on the mango
simulation clock and marking the behavior dirty so the next energy-flow solve
reflects the degraded state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


class FaultExecutor(ABC):
    @abstractmethod
    def inject_fault(self, net, time):
        ...

    @abstractmethod
    def reverse_fault(self, net, time):
        ...


class BranchOutage(FaultExecutor):
    """Deactivate a set of monee branches during the fault window."""

    def __init__(self, branch_ids) -> None:
        self._branch_ids = list(branch_ids)

    def inject_fault(self, net, time):
        for bid in self._branch_ids:
            net.branch_by_id(bid).active = False

    def reverse_fault(self, net, time):
        for bid in self._branch_ids:
            net.branch_by_id(bid).active = True


class Fault:
    def __init__(
        self, fault_executor: FaultExecutor, time_ranges: List[Tuple[int, int]]
    ) -> None:
        self._fault_executor = fault_executor
        self._start_time_steps = [tr[0] for tr in time_ranges]
        self._stop_time_steps = [tr[1] for tr in time_ranges]

    @property
    def fault_executor(self):
        return self._fault_executor

    @property
    def start_time_steps(self):
        return self._start_time_steps

    @property
    def stop_time_steps(self):
        return self._stop_time_steps


@dataclass
class _FaultState:
    faults: List[Fault]


def apply_faults_at_step(behavior, faults: List[Fault], time: int) -> None:
    """Inject / reverse any faults due at integer step ``time``; mark the behavior
    dirty if anything changed."""
    if not faults:
        return
    changed = False
    for fault in faults:
        if time in fault.start_time_steps:
            fault.fault_executor.inject_fault(behavior.net, time)
            changed = True
        if time in fault.stop_time_steps:
            fault.fault_executor.reverse_fault(behavior.net, time)
            changed = True
    if changed:
        behavior.mark_dirty()
