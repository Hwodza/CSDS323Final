import random
import numpy as np
from scipy.sparse import csr_matrix, diags

random.seed(0)


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
    pass


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
    H = np.zeros([len(nodelabels), 7])
    all_Z_histories = []
    final_Zs = []
    mse_history = []
    for rho in rhos:
        labels = perturb_H(rho, nodelabels)
        for i, label in enumerate(labels):
            H[i, label] = 1
            
        # Convert to sparse matrix
        H = csr_matrix(H)

        # Display sparse matrix
        # print(H)

        # --------------------Calculate Z-------------------------------
        z_history = []
        Z = H.copy()
        z_history.append(Z)
        alpha = .1
        K = 10

        # Iterative propagation
        for num in range(K):
            Z = (1 - alpha) * normalized_adj @ Z + alpha * H
            z_history.append(Z)
        
        all_Z_histories.append(z_history)

        # Apply softmax at the end
        Z = softmax(Z)
        final_Zs.append(Z)

        print(Z)
        err = compute_mse(Z, H)
        mse_history.append(err)  
        print("Error:", err, "\nAccuracy:", (1-err)*100, "%")


if __name__ == "__main__":
    main()