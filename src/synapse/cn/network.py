from peext.node import (
    EmptyBusNode,
    EmptyJunctionNode,
    RegulatableController,
    CouplingPoint,
)
from peext.edge import LineEdge, PipeEdge
from peext.network import get_bus_junc

import networkx as nx
import networkx.drawing.nx_agraph as nxd
import plotly.graph_objects as go


def gen_id(node):
    return f"{node.name}:{node.model}:{node.id}"


def gen_id_bus_junc(bus_junc):
    return f"{bus_junc[0]}:{bus_junc[1]}:{bus_junc[2]}"


def name_of(node):
    if isinstance(node, RegulatableController):
        return f"{gen_id(node)}:{type(node).__name__}"
    return gen_id(node)


def read_edge_length(edge):
    if isinstance(edge, (LineEdge, PipeEdge)):
        return edge.properties_as_dict()["length_km"]
    return 0


def is_node_drawable(node, only_include_active_elements):
    return (
        (
            not "in_service" in node.properties_as_dict()
            and not (
                isinstance(node, RegulatableController)
                and node.regulation_factor() != 0
            )
        )
        or node.properties_as_dict()["in_service"]
        or not only_include_active_elements
    )


def to_phys_bus_junc_networkx_graph(me_network, only_include_active_elements=True):
    graph = nx.MultiGraph()

    bus_junc_list = []
    for node in me_network.nodes:
        bus_junc_list = get_bus_junc(node)
        node_id = gen_id(node)
        # elements nodes
        if not isinstance(node, EmptyBusNode) and not isinstance(
            node, EmptyJunctionNode
        ):
            if is_node_drawable(node, only_include_active_elements):
                inner_nodes = {}
                if isinstance(node, CouplingPoint):
                    inner_nodes = node.controller.nodes
                graph.add_node(
                    node_id,
                    label=name_of(node).replace(":", "\n"),
                    inner_nodes=inner_nodes,
                )
        for bus_junc in bus_junc_list:
            bus_junc_id = gen_id_bus_junc(bus_junc)
            # bus/junction nodes
            graph.add_node(bus_junc_id, label=bus_junc_id.replace(":", "\n"))

            if not isinstance(node, EmptyBusNode) and not isinstance(
                node, EmptyJunctionNode
            ):
                if is_node_drawable(node, only_include_active_elements):
                    # connection bus/junction - node
                    graph.add_edge(node_id, bus_junc_id, weight=0)

    for network_name, all_edges_for_network in me_network.edges_as_dict.items():
        for edge in all_edges_for_network:
            to_nodes = [(node[0], node[2]) for node in edge.nodes if node[1] == "to"]
            from_nodes = [
                (node[0], node[2]) for node in edge.nodes if node[1] == "from"
            ]

            if len(to_nodes) == 0 or len(from_nodes) == 0:
                raise Exception(
                    f"{to_nodes}.{network_name}.{from_nodes}.{edge.id}.{edge.nodes}.{type(edge)}"
                )

            to_node = to_nodes[0]
            from_node = from_nodes[0]
            bus_junc_list_to = get_bus_junc(to_node[0])
            bus_junc_list_from = get_bus_junc(from_node[0])

            final_junc_bus_to = bus_junc_list_to[0]
            final_junc_bus_from = bus_junc_list_from[0]
            if len(bus_junc_list_from) > 1:
                final_junc_bus_from_list = [
                    jb for jb in bus_junc_list_from if jb[0] == network_name
                ]
                if len(final_junc_bus_from_list) > 1:
                    final_junc_bus_from = [
                        jb
                        for jb in final_junc_bus_from_list
                        if jb[3] in from_node[1]
                        or from_node[1] in ["junction", "bus", ""]
                    ][0]
                else:
                    final_junc_bus_from = final_junc_bus_from_list[0]
            if len(bus_junc_list_to) > 1:
                final_junc_bus_to_list = [
                    jb for jb in bus_junc_list_to if jb[0] == network_name
                ]
                if len(final_junc_bus_to_list) > 1:
                    final_junc_bus_to = [
                        jb
                        for jb in final_junc_bus_to_list
                        if jb[3] in to_node[1] or to_node[1] in ["junction", "bus", ""]
                    ][0]
                else:
                    final_junc_bus_to = final_junc_bus_to_list[0]

            if (
                edge.properties_as_dict()["in_service"]
                or not only_include_active_elements
            ):
                graph.add_edge(
                    gen_id_bus_junc(final_junc_bus_from),
                    gen_id_bus_junc(final_junc_bus_to),
                    weight=read_edge_length(edge),
                    label=f"{edge.network.name}:{edge.component_type()}:{edge.id}",
                )

    return graph


def create_networkx_topology_plot(graph, color_legend_text=None, title=None, pos=None):
    if pos is None:
        pos = nxd.pygraphviz_layout(graph, prog="neato")
    edge_x = []
    edge_y = []
    for from_node, to_node, _ in graph.edges:
        x0, y0 = pos[from_node]
        x1, y1 = pos[to_node]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x_power = []
    node_y_power = []
    node_text_power = []
    node_x_heat = []
    node_y_heat = []
    node_text_heat = []
    node_x_gas = []
    node_y_gas = []
    node_text_gas = []
    node_cp_x = []
    node_cp_y = []
    node_virt_x = []
    node_virt_y = []
    for node in graph.nodes:
        x, y = pos[node]
        if "bus" in node or "junction" in node:
            node_virt_x.append(x)
            node_virt_y.append(y)
        elif "controller" in node:
            node_cp_x.append(x)
            node_cp_y.append(y)
        else:
            if "heat" in node:
                node_text_heat.append(node)
                node_x_heat.append(x)
                node_y_heat.append(y)
            elif "gas" in node:
                node_text_gas.append(node)
                node_x_gas.append(x)
                node_y_gas.append(y)
            elif "power" in node:
                node_text_power.append(node)
                node_x_power.append(x)
                node_y_power.append(y)

    # cp
    node_trace_cp = go.Scatter(
        x=node_cp_x,
        y=node_cp_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color="purple",
            symbol="diamond",
            size=10,
            line=dict(width=2, color="#888"),
        ),
    )
    node_trace_virtual = go.Scatter(
        x=node_virt_x,
        y=node_virt_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color="black",
            size=10,
            line=dict(width=2, color="#888"),
        ),
    )

    # heat
    node_trace_heat = go.Scatter(
        x=node_x_heat,
        y=node_y_heat,
        mode="markers",
        hoverinfo="text",
        text=node_text_heat,
        marker=dict(
            symbol="pentagon",
            size=10,
            line=dict(width=2, color="#888"),
        ),
    )
    # power
    node_trace_power = go.Scatter(
        x=node_x_power,
        y=node_y_power,
        mode="markers",
        hoverinfo="text",
        text=node_text_power,
        marker=dict(
            symbol="square",
            size=10,
            line=dict(width=2, color="#888"),
        ),
    )
    # gas
    node_trace_gas = go.Scatter(
        x=node_x_gas,
        y=node_y_gas,
        mode="markers",
        hoverinfo="text",
        text=node_text_gas,
        marker=dict(
            symbol="triangle-up",
            size=10,
            line=dict(width=2, color="#888"),
        ),
    )

    fig = go.Figure(
        data=[
            edge_trace,
            node_trace_virtual,
            node_trace_heat,
            node_trace_power,
            node_trace_gas,
            node_trace_cp,
        ],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
        ),
    )
    fig.update_layout(
        margin={"l": 20, "b": 30, "r": 10, "t": 30},
        xaxis_title="",
        legend={"title": color_legend_text},
        yaxis_title="",
        title=title,
    )
    return fig
