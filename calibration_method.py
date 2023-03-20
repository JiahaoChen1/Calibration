#!/usr/bin/env python3
"""
Adapted and changed from: author: Jize Zhang
"""
from cmath import log
import numpy as np
from scipy import optimize
from sklearn.isotonic import IsotonicRegression
from scipy.special import softmax

"""
auxiliary functions for optimizing the temperature (scaling approaches) and weights
of ensembles
*args include logits and labels from the calibration dataset:
"""


def mse_t(t, *args):
    # find optimal temperature with MSE loss function
    logit, label = args
    logit = logit / t
    n = np.sum(np.clip(np.exp(logit), -1e20, 1e20), 1)
    p = np.clip(np.exp(logit), -1e20, 1e20) / n[:, None]
    mse = np.mean((p - label) ** 2)
    return mse


def ll_t(t, *args):
    # find optimal temperature with Cross-Entropy loss function

    logit, label = args
    logit = logit / t
    n = np.sum(np.clip(np.exp(logit), -1e20, 1e20), 1)
    p = np.clip(np.clip(np.exp(logit), -1e20, 1e20) / n[:, None], 1e-20, 1 - 1e-20)
    N = p.shape[0]
    ce = -np.sum(label * np.log(p)) / N
    return ce


def mse_w(w, *args):
    # find optimal weight coefficients with MSE loss function

    p0, p1, p2, label = args
    p = w[0] * p0 + w[1] * p1 + w[2] * p2
    p = p / np.sum(p, 1)[:, None]
    mse = np.mean((p - label) ** 2)
    return mse


def ll_w(w, *args):
    # find optimal weight coefficients with Cros-Entropy loss function

    p0, p1, p2, label = args
    p = (w[0] * p0 + w[1] * p1 + w[2] * p2)
    N = p.shape[0]
    ce = -np.sum(label * np.log(p)) / N
    return ce


def mse_w_entropy(w, *args):
    # find optimal weight coefficients with MSE loss function

    p0, p1, p2, p3, label = args
    p = w[0] * p0 + w[1] * p1 + w[2] * p2 + w[3] * p3
    p = p / np.sum(p, 1)[:, None]
    mse = np.mean((p - label) ** 2)
    return mse


def ll_w_entropy(w, *args):
    # find optimal weight coefficients with Cros-Entropy loss function

    p0, p1, p2, p3, label = args
    p = (w[0] * p0 + w[1] * p1 + w[2] * p2 + w[3] * p3)
    N = p.shape[0]
    ce = -np.sum(label * np.log(p)) / N
    return ce


def mse_w_ts_entropy(w, *args):
    logit, label, t, entropy = args
    entropy = np.resize(entropy, (np.shape(entropy)[0], 1))
    t = np.resize(t, (np.shape(entropy)[0], 1))
    t_ent = t + w[0] * (entropy - w[1])
    logit = logit / t_ent
    n = np.sum(np.clip(np.exp(logit), -1e20, 1e20), 1)
    p = np.clip(np.exp(logit), -1e20, 1e20) / n[:, None]
    mse = np.mean((p - label) ** 2)
    return mse


def ll_w_ts_entropy(w, *args):
    # find optimal weight coefficients with Cros-Entropy loss function
    logit, label, t, entropy = args
    t_ent = t + w[0] * (entropy - w[1])
    t_ent = np.resize(t_ent, (np.shape(t_ent)[0], 1))
    logit = logit / t_ent
    p = softmax(logit, axis=1)
    N = p.shape[0]
    ce = -np.sum(label * np.log(p)) / N
    return ce

###################
# Fitting function 
###################

# Ftting Temperature Scaling
def train_temperature_scaling(logit, label, loss):
    bnds = ((0.05, 5.0),)
    if loss == 'ce':
        t = optimize.minimize(
            ll_t,
            1.0,
            args=(logit, label),
            method='L-BFGS-B',
            bounds=bnds, tol=1e-12,
            options={'disp': False})
    if loss == 'mse':
        t = optimize.minimize(
            mse_t,
            1.0,
            args=(logit, label),
            method='L-BFGS-B',
            bounds=bnds,
            tol=1e-12,
            options={'disp': False})
    t = t.x
    print('temperature scaling {}'.format(t))
    return t


# Fitting Enseble Temperature Scaling
def util_ensemble_temperature_scaling(logit, label, t, n_class, loss):
    p1 = softmax(logit, axis=1)
    logit = logit / t
    p0 = softmax(logit, axis=1)
    p2 = np.ones_like(p0) / n_class

    bnds_w = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0),)

    def my_constraint_fun(x):
        return np.sum(x) - 1

    constraints = {"type": "eq", "fun": my_constraint_fun, }
    if loss == 'ce':
        w = optimize.minimize(
            ll_w,
            (1.0, 0.0, 0.0),
            args=(p0, p1, p2, label),
            method='SLSQP',
            constraints=constraints,
            bounds=bnds_w, tol=1e-12,
            options={'disp': False})
    if loss == 'mse':
        w = optimize.minimize(
            mse_w, (1.0, 0.0, 0.0),
            args=(p0, p1, p2, label),
            method='SLSQP',
            constraints=constraints,
            bounds=bnds_w, tol=1e-12,
            options={'disp': False})
    w = w.x
    return w


# Fitting Enseble Temperature Scaling
def util_ensemble_temperature_scaling_entropy(logit, label, t, n_class, loss):
    p1 = softmax(logit, axis=1)
    logit = logit / t
    p0 = softmax(logit, axis=1)
    p2 = np.ones_like(p0) / n_class
    entropy = np.abs(np.sum(np.multiply(p0, np.clip(np.log(p0), -1e20, 1e20)), axis=1))
    p3 = p2 * np.resize(entropy, (np.shape(entropy)[0], 1))

    bnds_w = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),)

    def my_constraint_fun(x):
        return np.sum(x) - 1

    constraints = {"type": "eq", "fun": my_constraint_fun, }
    if loss == 'ce':
        w = optimize.minimize(
            ll_w_entropy,
            (1.0, 0.0, 0.0, 0.0),
            args=(p0, p1, p2, p3, label),
            method='SLSQP',
            constraints=constraints,
            bounds=bnds_w, tol=1e-12,
            options={'disp': False})
    if loss == 'mse':
        w = optimize.minimize(
            mse_w_entropy,
            (1.0, 0.0, 0.0, 0.0),
            args=(p0, p1, p2, p3, label),
            method='SLSQP',
            constraints=constraints,
            bounds=bnds_w,
            tol=1e-12,
            options={'disp': False})
    w = w.x
    return w


def util_temperature_scaling_entropy(logit, label, t, n_class, loss):
    p = softmax(logit, axis=1)
    entropy = np.abs(np.sum(np.multiply(p, np.clip(np.log(p), -1e20, 1e20)), axis=1))
    entropy1 = np.ones((np.shape(entropy)[0])) * \
        np.random.normal(0.0, 0.000001, (np.shape(entropy)[0]))
    entropy = np.abs(entropy + entropy1)
    bnds_w = ((-100.0, 100.0), (0.0, 100.0), (0.0, 5.0),)

    def my_constraint_fun(x):
        return np.sum(x) - 1

    constraints = ()  # { "type":"eq", "fun":my_constraint_fun,}
    if loss == 'ce':
        w = optimize.minimize(
            ll_w_ts_entropy,
            (0.5, 0.0, 1.0),
            args=(logit, label, t, entropy),
            method='L-BFGS-B',
            constraints=constraints,
            bounds=bnds_w,
            tol=1e-12,
            options={'disp': False})
    if loss == 'mse':
        w = optimize.minimize(
            mse_w_ts_entropy,
            (1.0, 1.0, 1.0),
            args=(logit, label, t, entropy),
            method='L-BFGS-B',
            bounds=bnds_w,
            tol=1e-12,
            options={'disp': False})
    w = w.x
    return w



###################
# Train function 
###################

def train_ensemble_temperature_scaling(logit, label, n_class, loss):
    t = train_temperature_scaling(logit, label, loss)  # loss can be 'mse' or 'ce'
    w = util_ensemble_temperature_scaling(logit, label, t, n_class, loss)
    print(t, w)
    return (t, w)


def train_ensemble_temperature_scaling_entropy(logit, label, n_class, loss):
    t = train_temperature_scaling(logit, label, loss)  # loss can be 'mse' or 'ce'
    w = util_ensemble_temperature_scaling_entropy(logit, label, t, n_class, loss)
    return (t, w)


def train_temperature_scaling_entropy(logit, label, n_class, loss):
    t = train_temperature_scaling(logit, label, loss)  # loss can be 'mse' or 'ce'
    w = util_temperature_scaling_entropy(logit, label, t, n_class, loss)
    return (t, w)


# Fitting: Isotonic Regression (Multi-class)
def train_isotonic_regression(logits, labels):
    p = softmax(logits, axis=1)
    ir = IsotonicRegression(out_of_bounds='clip')
    y_ = ir.fit_transform(p.flatten(), (labels.flatten()))
    # ir_draw(p.flatten(), labels.flatten(), y_, 'fig/IRM_curve.jpg')
    return ir


def train_temperature_scaling_isotonic_regression(logits, labels, loss):
    t = train_temperature_scaling(logits, labels, loss)
    ir = train_isotonic_regression(logits, labels)
    return (t, ir)


# Fitting: Isotonic Regression One vs all
def train_irova(logits, labels):
    p = softmax(logits, axis=1)
    list_ir = []
    for ii in range(p.shape[1]):
        ir = IsotonicRegression(out_of_bounds='clip')
        y_ = ir.fit_transform(p[:, ii].astype('double'), labels[:, ii].astype('double'))
        list_ir.append(ir)
        # ir_draw(p[:, ii].astype('double'), labels[:, ii].astype('double'), y_, 'fig/cifar101_ir_curve/{}.jpg'.format(ii))
    return list_ir


# Fitting: Isotonic Regression One vs all + TS
def train_irovats(logits, labels, loss="mse"):
    t = train_temperature_scaling(logits, labels, loss=loss)
    logits = logits / t
    list_ir = train_irova(logits, labels)
    return (t, list_ir)


"""
Calibration:
Input: uncalibrated logits, temperature (and weight)
Output: calibrated prediction probabilities
Applies for all below functions
"""

# Calibration: Temperature Scaling
def calibrate_temperature_scaling(logits, t):
    logits = logits / t
    p = softmax(logits, axis=1)
    return p


# Calibration: Ensemble Temperature Scaling
def calibrate_ensemble_temperature_scaling(logits, t, w, n_class):
    p1 = softmax(logits, axis=1)
    logits = logits / t
    p0 = softmax(logits, axis=1)
    p2 = np.ones_like(p0) / n_class
    p = w[0] * p0 + w[1] * p1 + w[2] * p2
    return p


# Calibration: Ensemble Temperature Scaling - Entropy
def calibrate_ensemble_temperature_scaling_entropy(logits, t, w, n_class):
    p1 = softmax(logits, axis=1)
    logits = logits / t
    p0 = softmax(logits, axis=1)
    p2 = np.ones_like(p0) / n_class
    entropy = np.abs(np.sum(np.multiply(p0, np.clip(np.log(p0), -1e20, 1e20)), axis=1))
    p3 = p2 * np.resize(entropy, (np.shape(entropy)[0], 1))
    p = w[0] * p0 + w[1] * p1 + w[2] * p2 + w[3] * p3
    return p


# Calibration: Temperature Scaling - Entropy
def calibrate_temperature_scaling_entropy(logits, t, w, n_class):
    p0 = softmax(logits, axis=1)
    entropy = np.abs(np.sum(np.multiply(p0, np.clip(np.log(p0), -1e20, 1e20)), axis=1))
    t_ent = t + w[0] * (entropy - w[1])
    t_ent = np.resize(t_ent, (np.shape(t_ent)[0], 1))
    logits = logits / t_ent
    p = softmax(logits, axis=1)
    return p


# Calibration: Isotonic Regression (Multi-class)
def calibrate_isotonic_regression(logits, ir):
    p_eval = softmax(logits, axis=1)
    yt_ = ir.predict(p_eval.flatten())
    p = yt_.reshape(logits.shape) + 1e-9 * p_eval
    return p


def calibrate_temperature_scaling_isotonic_regression(logits, t, ir):
    logits = logits / t
    p = calibrate_isotonic_regression(logits, ir)
    return p


# Calibrate IROVA
def calibrate_irova(logits, list_ir):
    p_eval = softmax(logits, axis=1)
    for ii in range(p_eval.shape[1]):
        ir = list_ir[ii]
        p_eval[:, ii] = ir.predict(p_eval[:, ii]) + 1e-9 * p_eval[:, ii]
    return p_eval


# Calibrate IROVA + TS
def calibrate_irovats(logits, t, list_ir):
    logits = logits / t
    p_eval = calibrate_irova(logits, list_ir)
    return p_eval


###############
#ours
###############

def train_class_temperature_scaling(logits, labels, loss='ce'):
    bnds = [(0.05, 5.0) for _ in range(logits.shape[-1])]
    if loss == 'ce':
        t = optimize.minimize(
            ll_t,
            np.array([1.0 for _ in range(logits.shape[-1])]),
            args=(logits, labels),
            method='L-BFGS-B',
            bounds=bnds, tol=1e-12,
            options={'disp': False})
    elif loss == 'mse':
        t = optimize.minimize(
            mse_t,
            np.array([1.0 for _ in range(logits.shape[-1])]),
            args=(logits, labels),
            method='L-BFGS-B',
            bounds=bnds,
            tol=1e-12,
            options={'disp': False})
    t = t.x 
    return t

def calibrate_class_temperature_scaling(logits, t):
    logits = logits / t
    p = softmax(logits, axis=1)
    return p


def mse_t_group(t, *args):
    logits, labels, a = args
    mse = 0
    for i in range(logits.shape[0]):
        sample = logits[i]
        logit = sample[:, None] / t
        logit = logit[:, a[i]]
        n = np.sum(np.clip(np.exp(logit), -1e20, 1e20))
        p = np.clip(np.exp(logit), -1e20, 1e20) / n
        mse = mse + np.mean((p - labels[i]) ** 2)
    return mse / logits.shape[0]

def ll_t_group(t, *args):
    logits, labels, a = args
    ce = 0
    for i in range(logits.shape[0]):
        sample = logits[i]
        logit = sample[:, None] / t
        logit = logit[:, a[i]]
        n = np.sum(np.clip(np.exp(logit), -1e20, 1e20))
        p = np.clip(np.exp(logit), -1e20, 1e20) / n
        ce = ce - np.sum(labels[i] * np.log(p))
    return ce / logits.shape[0]

def convert(logits, group):
    a = np.zeros((logits.shape[0],))
    predicts = np.argmax(logits, axis=1)
    for  i in range(logits.shape[0]):
        for j  in  range(len(group)):
            if predicts[i]  in group[j]:
                a[i] = j
    return a.astype(np.int64)

def train_group_temperature_scaling(logits, labels, loss='mse', group=[]):
    bnds = [(0.05, 5.0) for _ in range(len(group))]
    a = convert(logits, group)
    if loss == 'ce':
        t = optimize.minimize(
            ll_t_group,
            np.array([1.0 for _ in range(len(group))]),
            args=(logits, labels, a),
            method='L-BFGS-B',
            bounds=bnds, tol=1e-12,
            options={'disp': False})
    elif loss == 'mse':
        t = optimize.minimize(
            mse_t_group,
            np.array([1.0 for _ in range(len(group))]),
            args=(logits, labels, a),
            method='L-BFGS-B',
            bounds=bnds,
            tol=1e-12,
            options={'disp': False})
    t = t.x 
    print(t)
    return t

def calibrate_group_temperature_scaling(logits, t, group=[]):
    a = convert(logits, group)
    p = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        p[i] = softmax(logits[i] / t[a[i]])
    return p


def mse_t_da(t, *args):
    logits, labels, w = args
    logits = logits / t
    n = np.sum(np.clip(np.exp(logits), -1e20, 1e20), 1)
    p = np.clip(np.exp(logits), -1e20, 1e20) / n[:, None]
    mse = np.mean(((p - labels) ** 2) * w[:, None])
    return mse

def ll_t_da(t, *args):
    logits, labels, w = args
    logits = logits / t
    n = np.sum(np.clip(np.exp(logits), -1e20, 1e20), 1)
    p = np.clip(np.clip(np.exp(logits), -1e20, 1e20) / n[:, None], 1e-20, 1 - 1e-20)
    N = p.shape[0]
    ce = -np.sum(labels * np.log(p) * w[:, None]) / N
    return ce

def train_da_temperature_scaling(logit, label, loss, w):
    bnds = ((0.05, 5.0),)
    if loss == 'ce':
        t = optimize.minimize(
            ll_t_da,
            1.0,
            args=(logit, label, w),
            method='L-BFGS-B',
            bounds=bnds, tol=1e-12,
            options={'disp': False})
    if loss == 'mse':
        t = optimize.minimize(
            mse_t_da,
            1.0,
            args=(logit, label, w),
            method='L-BFGS-B',
            bounds=bnds,
            tol=1e-12,
            options={'disp': False})
    t = t.x
    print('Temperature scaling {}'.format(t))
    return t