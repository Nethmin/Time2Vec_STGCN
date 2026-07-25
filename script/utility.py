import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch


def calc_gso(dir_adj, gso_type):
    """Build the graph shift operator. The final model uses 'sym_norm_lap'
    (symmetrically normalized Laplacian); the other variants are kept for
    completeness since they cost nothing."""
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrize the adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)

    if gso_type in ('sym_renorm_adj', 'rw_renorm_adj', 'sym_renorm_lap', 'rw_renorm_lap'):
        adj = adj + id  # add self-loops

    if gso_type in ('sym_norm_adj', 'sym_renorm_adj', 'sym_norm_lap', 'sym_renorm_lap'):
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_sym = D^-0.5 * A * D^-0.5
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type in ('sym_norm_lap', 'sym_renorm_lap'):
            gso = id - sym_norm_adj
        else:
            gso = sym_norm_adj

    elif gso_type in ('rw_norm_adj', 'rw_renorm_adj', 'rw_norm_lap', 'rw_renorm_lap'):
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_rw = D^-1 * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type in ('rw_norm_lap', 'rw_renorm_lap'):
            gso = id - rw_norm_adj
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso


def calc_chebynet_gso(gso):
    """Rescale the Laplacian to [-1, 1] (approximately) for Chebyshev polynomials."""
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    eigval_max = norm(gso, 2)

    # For a symmetric / random-walk normalized Laplacian the max eigenvalue <= 2.
    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id

    return gso


def evaluate_model(model, loss, data_iter):
    """Mean loss (same loss used in training) over a data iterator."""
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, tod, dow, y in data_iter:
            y_pred = model(x, tod, dow).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def evaluate_metric(model, data_iter, scaler, mask_value=1e-5):
    """Accuracy metrics on the original (inverse-transformed) scale.

    Returns (MAE, RMSE, WMAPE, MAPE):
      MAE   -- mean absolute error
      RMSE  -- root mean squared error
      WMAPE -- sum|err| / sum|y|  (robust to small denominators)
      MAPE  -- masked mean abs pct error; entries with |y| <= mask_value are
               excluded so near-zero speeds don't blow the metric up.
    """
    model.eval()
    abs_err_sum = 0.0
    sq_err_sum = 0.0
    y_abs_sum = 0.0
    n_total = 0
    mape_sum = 0.0
    mape_count = 0
    with torch.no_grad():
        for x, tod, dow, y in data_iter:
            y_data = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(
                model(x, tod, dow).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y_data - y_pred)

            abs_err_sum += float(d.sum())
            sq_err_sum += float(np.square(d).sum())
            y_abs_sum += float(np.abs(y_data).sum())
            n_total += d.size

            mask = np.abs(y_data) > mask_value
            mape_sum += float((d[mask] / np.abs(y_data[mask])).sum())
            mape_count += int(mask.sum())

    MAE = abs_err_sum / n_total
    RMSE = np.sqrt(sq_err_sum / n_total)
    WMAPE = abs_err_sum / y_abs_sum if y_abs_sum > 0 else float('nan')
    MAPE = mape_sum / mape_count if mape_count > 0 else float('nan')

    return MAE, RMSE, WMAPE, MAPE
