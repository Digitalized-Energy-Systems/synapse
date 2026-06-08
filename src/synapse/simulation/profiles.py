"""Demand profiles for the monee-based simulation.

The former pandapower ``ConstControl`` / simbench attachment is gone.  A profile
is now a :class:`DemandSchedule`: a per-element, per-step setpoint table that the
mango world applies onto monee child / branch models at every time step (and
marks the network dirty so the next energy-flow solve picks the new demand up).
"""

import logging

import numpy as np

import monee.model as mm

logger = logging.getLogger(__name__)


def create_random_profile(element_len, time_steps, center=0, dev=0.1, only_positive=False):
    matr = np.random.normal(center, dev, (time_steps, element_len))
    if only_positive:
        matr = np.abs(matr)
    return matr


class DemandSchedule:
    """A table of per-step setpoints.

    Entries are ``(kind, component_id, attr) -> np.ndarray[time_steps]`` where
    ``kind`` is ``"child"`` or ``"branch"``.  :meth:`apply` writes the step's
    values onto the live monee network and returns ``True`` when anything
    changed (so the world can mark itself dirty).
    """

    def __init__(self, time_steps):
        self.time_steps = time_steps
        self._entries = {}

    def add(self, kind, component_id, attr, values):
        self._entries[(kind, component_id, attr)] = np.asarray(values)

    def apply(self, net, t):
        changed = False
        for (kind, component_id, attr), values in self._entries.items():
            if t >= len(values):
                continue
            try:
                if kind == "child":
                    model = net.child_by_id(component_id).model
                else:
                    model = net.branch_by_id(component_id).model
            except Exception:
                continue
            setattr(model, attr, float(values[t]))
            changed = True
        return changed


def _base_value(model, attr):
    val = getattr(model, attr, None)
    if hasattr(val, "value"):
        val = val.value
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def random_demand_schedule(net, time_steps, dev=0.1, seed=None):
    """Build a :class:`DemandSchedule` fluctuating each demand around a small
    fraction of its base setpoint: power loads (``p_mw``), gas/water sinks
    (``mass_flow``) and heat exchangers (``q_mw_set``)."""
    if seed is not None:
        np.random.seed(seed)

    schedule = DemandSchedule(time_steps)

    loads = [c for c in net.childs if isinstance(c.model, mm.PowerLoad)]
    sinks = [c for c in net.childs if isinstance(c.model, mm.Sink)]
    hxs = [b for b in net.branches if isinstance(b.model, mm.HeatExchanger)]

    if loads:
        factor = create_random_profile(len(loads), time_steps, dev=dev, only_positive=True)
        for i, child in enumerate(loads):
            base = _base_value(child.model, "p_mw")
            schedule.add("child", child.id, "p_mw", factor[:, i] * base)
    if sinks:
        factor = create_random_profile(len(sinks), time_steps, dev=dev, only_positive=True)
        for i, child in enumerate(sinks):
            base = _base_value(child.model, "mass_flow")
            schedule.add("child", child.id, "mass_flow", factor[:, i] * base)
    if hxs:
        factor = create_random_profile(len(hxs), time_steps, dev=dev, only_positive=False)
        for i, branch in enumerate(hxs):
            attr = "q_mw_set" if hasattr(branch.model, "q_mw_set") else "q_mw"
            base = _base_value(branch.model, attr)
            schedule.add("branch", branch.id, attr, factor[:, i] * base)
    return schedule


def simbench_demand_schedule(net, timeseries_data, time_steps=None):
    """Build a :class:`DemandSchedule` from a monee
    :class:`~monee.simulation.timeseries.TimeseriesData` simbench profile.

    Matches series to monee children by name; series whose name does not resolve
    to a child are skipped (logged at debug level).
    """
    name_data = getattr(timeseries_data, "child_name_data", {})
    # Resolve child name -> id once.
    name_to_id = {}
    for child in net.childs:
        name = getattr(child, "name", None) or getattr(child.model, "name", None)
        if name is not None:
            name_to_id[name] = child.id

    if time_steps is None:
        time_steps = max(
            (len(series) for attrs in name_data.values() for series in attrs.values()),
            default=0,
        )
    schedule = DemandSchedule(time_steps)
    for name, attrs in name_data.items():
        cid = name_to_id.get(name)
        if cid is None:
            logger.debug("simbench profile %r does not map to a monee child", name)
            continue
        for attr, series in attrs.items():
            schedule.add("child", cid, attr, series)
    return schedule
