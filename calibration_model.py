from cmath import log
import scipy
import math
from scipy.stats import multivariate_normal
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from utils import Wasserstein

from calibration_method import \
    train_temperature_scaling, \
    train_ensemble_temperature_scaling, \
    train_isotonic_regression, \
    train_irova, \
    train_irovats, \
    train_da_temperature_scaling,\
    calibrate_temperature_scaling, \
    calibrate_ensemble_temperature_scaling, \
    calibrate_isotonic_regression, \
    calibrate_irova, \
    calibrate_irovats


class BaseModel:
    """
    Base class for all post-hoc calibration models.
    """

    def __init__(self, init_theta=[0], folder="Base"):
        self.theta = init_theta
        self.folder = folder

    def function(self, logits, theta=None):
        """
        Calculate probabilities. For the base model this is just an
        identity function
        """
        if theta is None:
            theta = self.theta
        return logits


class TSModel(BaseModel):
    """
    Base class extended for Temperature Scaling.
    """

    def __init__(self, init_theta=None, folder="TS"):
        super().__init__(init_theta=init_theta, folder=folder)

    def function(self, logits, theta=None):
        """
        Calculate transformed probabilities.
        :param logits: tensor containing the logits of a single perturbation
        :param theta: model parameters
        :return: transformed logits
        """
        if theta is None:
            theta = self.theta
        calibrated_probs = calibrate_temperature_scaling(logits, theta)
        return np.log(np.clip(calibrated_probs, 1e-20, 1 - 1e-20))

    def optimize(self, logits, labels):
        """
        Tune post-hoc calibration model.
        :param logits: tensor containing the logits of a single perturbation
        :param labels: tensor containing the ground truth labels
        """
        self.theta = train_temperature_scaling(logits, labels, loss='ce')


class ETSModel(BaseModel):
    """
    Base class extended for Ensemble Temperature Scaling.
    """

    def __init__(self, init_theta=None, folder="ETS"):
        super().__init__(init_theta=init_theta, folder=folder)

    def function(self, logits, theta=None):
        """
        Calculate transformed probabilities
        :param logits: tensor containing the logits of a single perturbation
        :param theta: model parameters
        :return: transformed logits
        """
        if theta is None:
            theta = self.theta
        t, w = self.theta
        calibrated_probs = calibrate_ensemble_temperature_scaling(
            logits,
            t,
            w,
            n_class=self.n_class
        )
        return np.log(np.clip(calibrated_probs, 1e-20, 1 - 1e-20))

    def optimize(self, logits, labels):
        """
        Tune post-hoc calibration model.
        :param logits: tensor containing the logits of a single perturbation
        :param labels: tensor containing the ground truth labels
        """
        labels_1d = np.argmax(labels, axis=1)
        self.n_class = np.max(labels_1d) + 1
        self.theta = train_ensemble_temperature_scaling(
            logits,
            labels,
            n_class=self.n_class,
            loss='mse'
        )


class IRMModel(BaseModel):
    """
    Base class extended for multiclass isotonic regression.
    """

    def __init__(self, init_theta=None, folder="IRM"):
        super().__init__(init_theta=init_theta, folder=folder)

    def function(self, logits, theta=None):
        """
        Calculate transformed probabilities
        :param logits: tensor containing the logits of a single perturbation
        :param theta: model parameters
        :return: transformed logits
        """
        if theta is None:
            theta = self.theta
        calibrated_probs = calibrate_isotonic_regression(logits, theta)
        return np.log(np.clip(calibrated_probs, 1e-20, 1 - 1e-20))

    def optimize(self, logits, labels):
        """
        Tune post-hoc calibration model.
        :param logits: tensor containing the logits of a single perturbation
        :param labels: tensor containing the ground truth labels
        """
        self.theta = train_isotonic_regression(logits, labels)


class IROVAModel(BaseModel):
    """
    Base class extended for IRovA.
    """

    # we simply want a new default argument - otherwise this is not required
    def __init__(self, init_theta=None, folder="IROVA"):
        super().__init__(init_theta=init_theta, folder=folder)

    def function(self, logits, theta=None):
        """
        Calculate transformed probabilities
        :param logits: tensor containing the logits of a single perturbation
        :param theta: model parameters
        :return: transformed logits
        """
        if theta is None:
            theta = self.theta
        calibrated_probs = calibrate_irova(logits, theta)
        return np.log(np.clip(calibrated_probs, 1e-20, 1 - 1e-20))

    def optimize(self, logits, labels):
        """
        Tune post-hoc calibration model.
        :param logits: tensor containing the logits of a single perturbation
        :param labels: tensor containing the ground truth labels
        """
        self.theta = train_irova(logits, labels)


class IROVATSModel(BaseModel):
    """
    Base class extended for IRovA+TS.
    """

    # we simply want a new default argument - otherwise this is unrequired
    def __init__(self, init_theta=None, folder="IROVATS"):
        super().__init__(init_theta=init_theta, folder=folder)

    def function(self, logits, theta=None):
        """
        Calculate transformed probabilities
        :param logits: tensor containing the logits of a single perturbation
        :param theta: model parameters
        :return: transformed logits
        """
        if theta is None:
            theta = self.theta
        t, list_ir = theta
        calibrated_probs = calibrate_irovats(logits, t, list_ir)
        return np.log(np.clip(calibrated_probs, 1e-20, 1 - 1e-20))

    def optimize(self, logits, labels):
        """
        Tune post-hoc calibration model.
        :param logits: tensor containing the logits of a single perturbation
        :param labels: tensor containing the ground truth labels
        """
        self.theta = train_irovats(logits, labels, loss='mse')


class HistogramBinning(BaseModel):
    """
    Histogram Binning as a calibration method. The bins are divided into equal lengths.
    
    The class contains two methods:
        - fit(probs, true), that should be used with validation data to train the calibration model.
        - predict(probs), this method is used to calibrate the confidences.
    """
    
    def __init__(self, M=10):
        """
        M (int): the number of equal-length bins used
        """
        self.bin_size = 1./M  # Calculate bin size
        self.conf = []  # Initiate confidence list
        self.upper_bounds = np.arange(self.bin_size, 1+self.bin_size, self.bin_size)  # Set bin bounds for intervals

    
    def _get_conf(self, conf_thresh_lower, conf_thresh_upper, probs, true):
        """
        Inner method to calculate optimal confidence for certain probability range
        
        Params:
            - conf_thresh_lower (float): start of the interval (not included)
            - conf_thresh_upper (float): end of the interval (included)
            - probs : list of probabilities.
            - true : list with true labels, where 1 is positive class and 0 is negative).
        """

        # Filter labels within probability range
        filtered = [x[0] for x in zip(true, probs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
        nr_elems = len(filtered)  # Number of elements in the list.

        if nr_elems < 1:
            return 0
        else:
            # In essence the confidence equals to the average accuracy of a bin
            conf = sum(filtered)/nr_elems  # Sums positive classes
            return conf
    

    def optimize(self, logits, labels):
        """
        Fit the calibration model, finding optimal confidences for all the bins.
        
        Params:
            probs: probabilities of data
            true: true labels of data
        """
        self.m = []
        for k in range(logits.shape[-1]):
            conf = []
            probs = softmax(logits, axis=1)[:, k]
            # true = np.array(labels == k, dtype="int")
            true = labels[:, k]
            for conf_thresh in self.upper_bounds:
                temp_conf = self._get_conf((conf_thresh - self.bin_size), conf_thresh, probs = probs, true = true)
                conf.append(temp_conf)
            self.m.append(conf)


    # Fit based on predicted confidence
    def function(self, logits):
        """
        Calibrate the confidences
        
        Param:
            probs: probabilities of the data (shape [samples, classes])
            
        Returns:
            Calibrated probabilities (shape [samples, classes])
        """

        # Go through all the probs and check what confidence is suitable for it.
        probs = softmax(logits, axis=1)

        t = softmax(logits, axis=1)

        for k in range(logits.shape[-1]):
            for i, prob in enumerate(t[:, k]):
                idx = np.searchsorted(self.upper_bounds, prob)
                probs[i, k] = self.m[k][idx]

        probs = np.log(np.clip(probs, 1e-20, 1 - 1e-20))
        return probs


class ImportanceWeightsTS(BaseModel):
    def __init__(self, 
                 init_theta=[0],
                 folder="IWTS", 
                 features=None, 
                 logits=None, 
                 labels=None, 
                 train_features=None, 
                 train_labels=None, 
                 num_head_cls=4,
                 alpha=0.9):
        super().__init__(init_theta, folder)

        head_mask = np.array([-100., -100., -100., -100., -100., -100., -100., -100., -100., -100.])
        head_mask[:num_head_cls] = 1
        self.w = np.zeros((logits.shape[0]))

        dst_means, dst_covs = [], []
        target_norms = []
        for i in range(logits.shape[-1]):
            t_datas = train_features[train_labels == i, :]
            dst_means.append(np.mean(t_datas, axis=0))
            dst_covs.append(np.var(t_datas, axis=0))
            target_norms.append(multivariate_normal(mean=dst_means[-1], cov=dst_covs[-1], allow_singular=False))

        self.wasser_matrix = np.zeros((logits.shape[-1], logits.shape[-1]))
        for i in range(logits.shape[-1]):
            for j in range(logits.shape[-1]):
                if i == j:
                    self.wasser_matrix[i, j] = -1e9
                elif head_mask[j] == -100:
                    self.wasser_matrix[i, j] = -1e9
                else:
                    self.wasser_matrix[i, j] = -(Wasserstein(dst_means[i], dst_covs[i], dst_means[j], dst_covs[j])) / (train_features.shape[-1] ** (1/2))

            self.wasser_matrix[i] = softmax(self.wasser_matrix[i])

        for i in range(logits.shape[0]):
            gt_cls = labels[i]
            if head_mask[gt_cls] == 1.:
                self.w[i] = 1.
            else:
                shift_mean = np.sum(np.array(dst_means) * self.wasser_matrix[gt_cls][:, None], axis=0) * (1 - alpha) + dst_means[gt_cls] * alpha
                shift_cov = (np.sum(np.sqrt(np.array(dst_covs)) * self.wasser_matrix[gt_cls][:, None], axis=0) * (1 - alpha) + np.sqrt(dst_covs[gt_cls]) * alpha)** 2
                self.w[i] = np.exp(multivariate_normal(mean=shift_mean, cov=shift_cov, allow_singular=False).logpdf(features[i]) -\
                            target_norms[gt_cls].logpdf(features[i]))
                
                self.w[i] = np.clip(self.w[i], 0.3, 5)    

        
    def function(self, logits, theta=None):
        if theta is None:
            theta = self.theta
        calibrated_probs = calibrate_temperature_scaling(logits, theta)
        return np.log(np.clip(calibrated_probs, 1e-20, 1 - 1e-20))
    
    def optimize(self, logits, labels):
        self.theta = train_da_temperature_scaling(logits, labels, loss='ce', w=self.w)
