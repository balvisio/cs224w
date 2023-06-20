import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, y, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        print(f"edge Index:\n{edge_index}")
        # Step 2: Linearly transform node feature matrix.
        # x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=(y, y), norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, x_i, norm, bobobo=4):
        # x_j has shape [E, out_channels]
        print("=========")
        print(f"x_i: {x_i}")
        print(f"x_j: {x_j}")
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    
    
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[3, 11, 11, 11], [4, 12, 12, 12], [5, 13, 13, 13]], dtype=torch.float)
y = torch.tensor([[7, 17, 17, 17], [8, 18, 18, 18], [9, 19, 19, 19]], dtype=torch.float)


edge_index=edge_index.t().contiguous()
conv = GCNConv(4, 1)
conv(x, y, edge_index)
