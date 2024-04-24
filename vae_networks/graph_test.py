import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from load_data import DataHandler  # Make sure this module is properly defined
import plotly.graph_objects as go

filename = 'current_train/XDATCAR'
data = DataHandler(filename, 1, flatten=False)

k = 2

knn_data = []
for sample in data.data:
    t = torch.tensor(sample, dtype=torch.float)
    edge_index = knn_graph(t, k, loop=False)

    # Create a PyTorch Geometric Data object
    data = Data(x=t, edge_index=edge_index)
    knn_data.append(data)

print(knn_data[0])

def visualize(data):
    # Extract coordinates
    x, y, z = data.x[:, 0].numpy(), data.x[:, 1].numpy(), data.x[:, 2].numpy()

    # Prepare a list for edge traces
    edge_trace = []
    for edge in data.edge_index.t().numpy():
        x0, y0, z0 = data.x[edge[0]]
        x1, y1, z1 = data.x[edge[1]]
        edge_trace.append(go.Scatter3d(x=[x0, x1, None], y=[y0, y1, None], z=[z0, z1, None],
                                       mode='lines', line=dict(width=2, color='black')))

    # Node trace for all but the last 7 nodes
    node_trace = go.Scatter3d(x=x[:-7], y=y[:-7], z=z[:-7], 
                              mode='markers', 
                              marker=dict(size=5, color='blue', opacity=0.8))

    # Node trace for the last 7 nodes, highlighted in a different color, e.g., red
    highlight_node_trace = go.Scatter3d(x=x[-7:], y=y[-7:], z=z[-7:], 
                                        mode='markers', 
                                        marker=dict(size=5, color='red', opacity=0.8))

    # Create the figure by combining edge traces with both node traces
    fig = go.Figure(data=edge_trace + [node_trace, highlight_node_trace])
    # fig = go.Figure(data=[highlight_node_trace])

    # Set figure layout
    fig.update_layout(template="plotly_white", title="3D Graph Visualization")
    fig.show()

# visualize(knn_data[0])
