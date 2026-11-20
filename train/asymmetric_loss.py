"""Binary asymmetric loss derivatives with respect to XGBoost margins."""

import numpy as np
from scipy.special import expit


def asl_derivatives(margins, labels, gamma_pos, gamma_neg, clip):
    """Differentiate ASL on either side of the negative clipping boundary."""
    margins = np.asarray(margins, dtype=np.float64)
    positive = labels == 1
    p, not_p = expit(margins), expit(-margins)
    slope = p * not_p
    active = positive | (p > clip)
    gamma = np.where(positive, gamma_pos, gamma_neg)
    error = np.where(positive, not_p, np.maximum(p - clip, 0))
    error_grad = np.where(positive, -slope, np.where(active, slope, 0))

    if clip:
        q = np.minimum(not_p + clip, 1)
        neg_loss = -np.log(q)
        neg_grad = error_grad * (~positive) / q
        neg_hess = neg_grad * (1 - 2 * p + neg_grad)
    else:
        neg_loss = np.logaddexp(0, margins)
        neg_grad, neg_hess = p, slope
    log_loss = np.where(positive, np.logaddexp(0, -margins), neg_loss)
    log_grad = np.where(positive, -not_p, neg_grad)
    log_hess = np.where(positive, slope, neg_hess)

    # Apply the product rule to error**gamma times the log loss.
    ratio = error_grad / np.maximum(error, np.finfo(np.float64).tiny)
    weight = error ** gamma
    weight_grad = weight * gamma * ratio
    weight_hess = weight_grad * ((gamma - 1) * ratio + 1 - 2 * p)
    gradient = weight * log_grad + weight_grad * log_loss
    hessian = weight * log_hess + 2 * weight_grad * log_grad + weight_hess * log_loss
    return np.where(active, gradient, 0), np.where(active, hessian, 0)


def asl_objective(margins, matrix, gamma_pos, gamma_neg, clip):
    gradient, hessian = asl_derivatives(margins, matrix.get_label(), gamma_pos, gamma_neg, clip)
    # ASL is non-convex. Keep a small positive curvature for active examples.
    hessian = np.where(gradient != 0, np.maximum(hessian, 1e-6), 0)
    return gradient, hessian
