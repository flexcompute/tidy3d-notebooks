"""Utility routines for functional-style optimization in the tutorial notebooks.

The helpers here avoid mutating inputs so they play nicely with autograd.
"""

import autograd.numpy as np
from autograd.misc import flatten


def clip_params(params, bounds):
    """Clip a parameter dictionary according to per-key bounds.

    Parameters
    ----------
    params : dict[str, np.ndarray]
        Dictionary mapping parameter names to array values.
    bounds : dict[str, tuple[float | None, float | None]]
        Lower and upper limits for each parameter. Missing keys default to no
        clipping. ``None`` disables a bound on that side.

    Returns
    -------
    dict[str, np.ndarray]
        New dictionary with values clipped to the requested interval.
    """
    clipped = {}
    for key, value in params.items():
        lo, hi = bounds.get(key, (None, None))
        lo_val = -np.inf if lo is None else lo
        hi_val = np.inf if hi is None else hi
        clipped[key] = np.clip(value, lo_val, hi_val)
    return clipped


def _flatten(tree):
    """Return a flat representation of a pytree and its inverse transform."""
    flat, unflatten = flatten(tree)
    return np.array(flat, dtype=float), unflatten


def init_adam(params, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
    """Initialize Adam optimizer state for a parameter pytree.

    Parameters
    ----------
    params : dict[str, np.ndarray]
        Current parameter values used to size the optimizer state.
    lr : float = 1e-2
        Learning rate applied to each step.
    beta1 : float = 0.9
        Exponential decay applied to the first moment estimate.
    beta2 : float = 0.999
        Exponential decay applied to the second moment estimate.
    eps : float = 1e-8
        Numerical stabilizer added inside the square-root denominator.

    Returns
    -------
    dict[str, object]
        Dictionary holding the Adam accumulator vectors and hyperparameters.
    """
    flat_params, unflatten = _flatten(params)
    state = {
        "t": 0,
        "m": np.zeros_like(flat_params),
        "v": np.zeros_like(flat_params),
        "unflatten": unflatten,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
    }
    return state


def adam_update(grads, state):
    """Compute Adam parameter updates from gradients and state.

    Parameters
    ----------
    grads : dict[str, np.ndarray]
        Gradient pytree with the same structure as the parameters.
    state : dict[str, object]
        Optimizer state returned by :func:`init_adam`.

    Returns
    -------
    updates : dict[str, np.ndarray]
        Parameter deltas that should be subtracted from the current values.
    new_state : dict[str, object]
        Updated optimiser state after incorporating the gradients.
    """
    g_flat, _ = _flatten(grads)
    t = state["t"] + 1

    beta1 = state["beta1"]
    beta2 = state["beta2"]
    m = (1 - beta1) * g_flat + beta1 * state["m"]
    v = (1 - beta2) * (g_flat * g_flat) + beta2 * state["v"]

    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    updates_flat = state["lr"] * (m_hat / (np.sqrt(v_hat) + state["eps"]))

    new_state = {
        **state,
        "t": t,
        "m": m,
        "v": v,
    }
    updates = state["unflatten"](updates_flat)
    return updates, new_state


def apply_updates(params, updates):
    """Apply additive updates to a parameter pytree.

    Parameters
    ----------
    params : dict[str, np.ndarray]
        Original parameter dictionary.
    updates : dict[str, np.ndarray]
        Update dictionary produced by :func:`adam_update`.

    Returns
    -------
    dict[str, np.ndarray]
        New dictionary with ``updates`` subtracted element-wise.
    """
    p_flat, unflatten = _flatten(params)
    u_flat, _ = _flatten(updates)
    return unflatten(p_flat - u_flat)
