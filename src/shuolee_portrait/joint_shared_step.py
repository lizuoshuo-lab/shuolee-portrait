"""Linearized trust step for a shared latent and row-local compensation."""

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix


def row_constraints(jacobian, supported, shared_width):
    """Pack one common block and a separate compensation block per frame."""
    owners, metrics = np.nonzero(supported)
    normals = jacobian[owners, metrics].astype(np.float64, copy=False)
    row_count, width = normals.shape
    local_width = width-shared_width
    columns = np.concatenate((
        np.broadcast_to(np.arange(shared_width), (row_count, shared_width)),
        shared_width+owners[:, None]*local_width+np.arange(local_width),
    ), axis=1)
    matrix = csr_matrix((normals.ravel(), (np.repeat(np.arange(row_count), width), columns.ravel())),
                        shape=(row_count, shared_width+len(supported)*local_width))
    norm = np.linalg.norm(normals, axis=1)
    return matrix, owners, metrics, norm


def project_halfspaces(target, matrix, slack):
    """Project a step onto unit-normal halfspaces with a bounded dual solve."""
    if matrix.shape[0] == 0:
        return target.copy(), dict(constraints=0, dual_iterations=0, primal_residual=0.,
                                  dual_residual=0., certified=True)
    transpose = matrix.T.tocsr()
    linear = np.asarray(matrix @ target).ravel()-slack

    def dual(value):
        normal = transpose @ value
        return .5*float(normal @ normal)-float(value @ linear), matrix @ normal-linear

    result = minimize(dual, np.zeros(len(slack)), jac=True, method='L-BFGS-B',
                      bounds=[(0., None)]*len(slack),
                      options=dict(maxiter=192, maxls=30, ftol=1e-14, gtol=1e-9))
    projected = target-transpose @ result.x
    violation = matrix @ projected-slack
    primal = float(np.maximum(violation, 0.).max())
    dual_residual = float(np.abs(result.x-np.maximum(result.x+violation, 0.)).max())
    if not np.isfinite(projected).all() or not np.isfinite(dual_residual):
        raise RuntimeError('The shared step projection is nonfinite.')
    return projected, dict(constraints=len(slack), dual_iterations=int(result.nit),
                           primal_residual=primal, dual_residual=dual_residual,
                           certified=max(primal, dual_residual) <= 2e-5)
