"""Coupled-network topology helpers, monee-based.

The former peext bus/junction projection is replaced by a direct projection of
the monee network graph: monee already keeps nodes and branches in a networkx
``MultiGraph`` (``Network._network_internal``), so the physical topology used for
centrality / plotting is derived straight from it.
"""

import networkx as nx
import plotly.graph_objects as go


def _grid_name(node):
    grid = getattr(node, "grid", None)
    name = getattr(grid, "name", None)
    if name:
        return name
    return type(grid).__name__ if grid is not None else "unknown"


def _model_name(component):
    return type(component.model).__name__


def gen_id(node):
    return f"{_grid_name(node)}:{_model_name(node)}:{node.id}"


def to_phys_networkx_graph(monee_net, only_include_active_elements=True):
    """Project a monee network onto a weighted ``networkx.MultiGraph``.

    Node ids are monee node ids; each node carries ``grid`` / ``model`` /
    ``label`` attributes.  Edge weights are branch lengths where available,
    so shortest-path based metrics reflect physical distance.
    """
    graph = nx.MultiGraph()
    for node in monee_net.nodes:
        if only_include_active_elements and not getattr(node, "active", True):
            continue
        graph.add_node(
            node.id,
            grid=_grid_name(node),
            model=_model_name(node),
            label=gen_id(node),
        )
    for branch in monee_net.branches:
        if only_include_active_elements and not getattr(branch, "active", True):
            continue
        u, v = branch.id[0], branch.id[1]
        if not graph.has_node(u) or not graph.has_node(v):
            continue
        length = float(getattr(branch.model, "length_m", 0.0) or 0.0)
        graph.add_edge(u, v, weight=length, label=_model_name(branch))
    return graph


def betweenness_centrality(monee_net):
    """Betweenness centrality of the monee physical topology, keyed by node id."""
    return dict(nx.betweenness_centrality(nx.Graph(to_phys_networkx_graph(monee_net))))


def create_networkx_topology_plot(graph, title=None, pos=None):
    """Render a projected topology ``graph`` (from :func:`to_phys_networkx_graph`)
    with plotly.  ``pos`` may be a precomputed layout dict."""
    if pos is None:
        pos = nx.spring_layout(graph)
    edge_x = []
    edge_y = []
    for from_node, to_node, _ in graph.edges:
        x0, y0 = pos[from_node]
        x1, y1 = pos[to_node]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"),
        hoverinfo="none", mode="lines",
    )
    node_x, node_y, node_text = [], [], []
    for node in graph.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(graph.nodes[node].get("label", str(node)))
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers", hoverinfo="text", text=node_text,
        marker=dict(size=10, line=dict(width=2, color="#888")),
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title, showlegend=False, hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
        ),
    )
    return fig
