import random
import numpy as np
from scipy.sparse import csr_matrix, diags
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

random.seed(0)


# class GCNLayer(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_features, out_features)

#     def forward(self, X, A_hat):
#         out = torch.spmm(A_hat, X)
#         out = self.linear(out)
#         return out


# class GCN(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features):
#         super(GCN, self).__init__()
#         self.gcn1 = GCNLayer(in_features, hidden_features)
#         self.gcn2 = GCNLayer(hidden_features, out_features)

#     def forward(self, X, A_hat):
#         out = self.gcn1(X, A_hat)
#         out = F.relu(out)
#         out = self.gcn2(out, A_hat)
#         return out


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def simulate_features(node_labels):
    """
    Simulate features using one-hot encoding of node labels.

    Parameters:
        node_labels (ndarray): Array of node labels.

    Returns:
        ndarray: One-hot encoded feature matrix.
    """
    num_nodes = len(node_labels)
    num_classes = len(np.unique(node_labels))
    features = np.zeros((num_nodes, num_classes))
    features[np.arange(num_nodes), node_labels] = 1
    return features


def softmax(X):
    """
    Compute the softmax of each row of the matrix X, supporting both sparse and dense inputs.

    Parameters:
        X (csr_matrix or ndarray): Input matrix (n_nodes x n_classes).

    Returns:
        csr_matrix or ndarray: Matrix with softmax applied row-wise.
    """
    if isinstance(X, csr_matrix):
        # Sparse matrix handling
        X_dense = X.toarray()  # Convert sparse matrix to dense
        exp_X = np.exp(X_dense - np.max(X_dense, axis=1, keepdims=True))  # Stability
        row_sums = np.sum(exp_X, axis=1, keepdims=True)
        softmax_dense = exp_X / row_sums
        return csr_matrix(softmax_dense)  # Convert back to sparse
    else:
        # Dense matrix handling
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  # Stability
        row_sums = np.sum(exp_X, axis=1, keepdims=True)
        return exp_X / row_sums


# ---------Error Calculation---------

def compute_mse(Z, Y):
    """
    Compute the mean squared error (MSE) between predictions and ground truth.

    Parameters:
        Z (ndarray or csr_matrix): Predicted probabilities (n_nodes x n_classes).
        Y (ndarray or csr_matrix): Ground truth labels (n_nodes x n_classes).

    Returns:
        float: Mean squared error.
    """
    if hasattr(Z, "toarray"):  # Check if Z is sparse
        Z = Z.toarray()
    if hasattr(Y, "toarray"):  # Check if Y is sparse
        Y = Y.toarray()

    mse = np.mean((Z - Y) ** 2)  # Element-wise difference squared and mean
    return mse


def perturb_H(rho, node_labels):
    for i, label in enumerate(node_labels):
        random_int = random.randint(0, rho)
        node_labels[i] = (label + random_int) % 7
    return node_labels


def APPNP(H, A, alpha, K, eps):
    z_history = []
    Z = H.copy()
    z_history.append(Z)

    # Iterative propagation
    for num in range(K):
        Z = (1 - alpha) * A @ Z + alpha * H
        z_history.append(Z)
        if (abs(Z-z_history[-2]).toarray() < eps).all():
            print("Converged at iteration", num)
            break
    print("Final iteration:", K)
    return Z, z_history


def one_hot_encode(labels, num_classes):
    """
    One-hot encode the labels.

    Parameters:
        labels (ndarray): Array of labels.
        num_classes (int): Number of classes.

    Returns:
        ndarray: One-hot encoded labels.
    """
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data, criterion):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
    return loss.item()


def main():
    #-------------------Get A~^--------------------------------
    filepath_graph = 'graph.csv'
    example_edges = np.genfromtxt(filepath_graph, delimiter=',', dtype=int)

    # Deduplicate edges (ensure one direction for each undirected edge)
    unique_edges = set(tuple(sorted(edge)) for edge in example_edges)

    # Convert back to an array for processing
    unique_edges = np.array(list(unique_edges))

    # Create node-to-index mapping
    example_nodes = np.unique(unique_edges.flatten())
    node_to_index_example = {node: idx for idx, node in enumerate(example_nodes)}
    index_to_node_example = np.array(example_nodes)

    # Map edges to indices
    rows_example = np.array([node_to_index_example[edge[0]] for edge in unique_edges])
    cols_example = np.array([node_to_index_example[edge[1]] for edge in unique_edges])

    # Construct adjacency matrix in CSR format
    data_example = np.ones(len(unique_edges), dtype=int)
    num_nodes_example = len(example_nodes)
    adjacency_matrix_csr = csr_matrix(
        (data_example, (rows_example, cols_example)), shape=(num_nodes_example, num_nodes_example)
    )

    print(adjacency_matrix_csr)

    # Add self-loops 
    adj_with_self_loops = adjacency_matrix_csr + diags([1] * adjacency_matrix_csr.shape[0]) 
    # Compute degree matrix 
    degrees = np.array(adj_with_self_loops.sum(axis=1)).flatten() 
    degree_inv_sqrt = 1.0 / np.sqrt(degrees)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0  
    # Handle divide-by-zero issues 
    degree_matrix_inv_sqrt = diags(degree_inv_sqrt)
    # Compute symmetrically normalized adjacency matrix 
    normalized_adj = degree_matrix_inv_sqrt @ adj_with_self_loops @ degree_matrix_inv_sqrt 

    # A~^
    print(normalized_adj)

    # ----------------------- Get H -----------------------
    filepath_nodelabels = 'nodelabels.csv'

    nodelabels = np.genfromtxt(filepath_nodelabels, dtype=int)
    rhos = [0, 1, 2, 3, 4, 5, 6, 7]
    init_H = np.zeros([len(nodelabels), 7])
    H = init_H.copy()
    all_Z_histories = []
    final_Zs = []
    APPNP_mse_history = []
    GCN_mse_history = []
    filepath_graph = 'graph.csv'
    example_edges = np.genfromtxt(filepath_graph, delimiter=',', dtype=int)

    # Create edge index
    edge_index = torch.tensor(example_edges.T, dtype=torch.long)

    for rho in rhos:
        labels = perturb_H(rho, nodelabels)
        features = simulate_features(labels)
        X = torch.FloatTensor(features)
        Y = torch.FloatTensor(one_hot_encode(labels, features.shape[1]))

        gcn = GCN(in_channels=features.shape[1], hidden_channels=16, out_channels=7)
        sparse_normalized_adj = csr_matrix(normalized_adj)
        A_hat = torch.FloatTensor(sparse_normalized_adj.toarray())
        # Create a PyTorch Geometric data object
        data = Data(x=X, edge_index=edge_index, y=Y)
        model = GCN(in_channels=features.shape[1], hidden_channels=16, out_channels=features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        # Forward pass
        for epoch in range(200):  # Adjust the number of epochs as needed
            train_loss = train(model, data, optimizer, criterion)

        # Evaluate the model
        mse = test(model, data, criterion)
        GCN_mse_history.append(mse)
        print(f"Perturbation: {rho}, Mean Squared Error: {mse}")

        # Calculate MSE
        # mse = compute_mse(output.detach().numpy(), one_hot_encode(nodelabels, 7))
        # GCN_mse_history.append(mse)
        print("Mean Squared Error:", mse)
        for i, label in enumerate(labels):
            H[i, label] = 1
            
        # Convert to sparse matrix
        H = csr_matrix(H)
        if rho == 0:
            init_H = H.copy()

        # Display sparse matrix
        # print(H)

        # --------------------Calculate Z-------------------------------
        Z, z_history = APPNP(H, normalized_adj, alpha=0.1, K=100, eps=1e-11)
        
        all_Z_histories.append(z_history)

        # Apply softmax at the end
        Z = softmax(Z)
        final_Zs.append(Z)

        print(Z)
        err = compute_mse(Z, init_H)
        APPNP_mse_history.append(err)  
        print("Error:", err, "\nAccuracy:", (1-err)*100, "%")
    print("----------------------------")
    # print("Final Zs:")
    # print(final_Zs)
    # plt.figure(figsize=(12, 6))

    # Subplot for APPNP MSE history
    plt.plot(APPNP_mse_history)
    plt.xlabel('Possible Perturbations')
    plt.ylabel('Mean Squared Error')
    plt.title('APPNP MSE History')
    plt.show()

    # Subplot for GCN MSE history
    plt.plot(GCN_mse_history)
    plt.xlabel('Possible Perturbations')
    plt.ylabel('Mean Squared Error')
    plt.title('GCN MSE History')
    plt.show()


    




if __name__ == "__main__":
    main()