from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
from pandapower.control.basic_controller import Controller

class FaultExecutor(ABC):

    @abstractmethod
    def inject_fault(self, multinet, time):
        pass

    @abstractmethod
    def reverse_fault(self, multinet, time):
        pass

class Fault:

    def __init__(self, fault_executor: FaultExecutor, time_ranges: List[Tuple[int, int]]) -> None:
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

class FaultInjector(Controller):
    """Interface to the pandapipes/power controller system to overcome the need to have
    a mango agent. Useful for time-series simulations without the need of communication
    between real agents.
    """
    def __init__(self, multinet, faults: List[Fault], in_service=True, order=0,
                 level=0, drop_same_existing_ctrl=False, initial_run=True, **kwargs):
        super().__init__(multinet, in_service, order, level,
                        drop_same_existing_ctrl=drop_same_existing_ctrl, initial_run=initial_run,
                        **kwargs)

        self._faults = faults
        self._names = multinet['nets'].keys()

    def initialize_control(self, _):
        self.applied = False

    def get_all_net_names(self):
        return self._names

    def time_step(self, mn, time):
        
        if self._faults is not None:
            for fault in self._faults:
                if time in fault.start_time_steps:
                    fault.fault_executor.inject_fault(mn, time)
                if time in fault.stop_time_steps:
                    fault.fault_executor.reverse_fault(mn, time)

    def control_step(self, _):
        self.applied = True

    def is_converged(self, _):
        return self.applied