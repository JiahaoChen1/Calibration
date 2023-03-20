import argparse
import numpy as np

from utils import label2onehot
from calibration_metric import *
from calibration_model import *


def main(args):
    cifar10_z, cifar10_label = np.load('data/cifar10_val_z.npy'), np.load('data/cifar10_val_label.npy')
    logits, labels = cifar10_z[0], label2onehot(cifar10_z[0], cifar10_label[0])
    cifar10_feature = np.load('data/cifar10_val_feature.npy')

    train_feature = np.load('data/cifar10_train_feature.npy')
    train_label = np.load('data/cifar10_train_label.npy')
    # cls_number = class_instance_number(cifar10_label[0], cifar10_z.shape[-1])

    calibrated_model = ImportanceWeightsTS(features=cifar10_feature[0], 
                                           logits=cifar10_z[0], 
                                           labels=cifar10_label[0],
                                           train_features=train_feature[0], 
                                           train_labels=train_label[0], 
                                           num_head_cls=args.num_head_cls,
                                           alpha=args.alpha)
    calibrated_metric = ECELoss()
    calibrated_model.optimize(logits, labels)

    print('*************cifar10************')
    cifar10_logits, cifar10_labels = np.load('data/cifar10_test_z.npy'), np.load('data/cifar10_test_label.npy')

    cal_logits = calibrated_model.function(cifar10_logits[0])
    loss = calibrated_metric.loss(cal_logits, cifar10_labels[0])
    print(loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_head_cls", help="The number of head classes", type=int, default=4)
    parser.add_argument("--alpha", help="hyper-parameter", type=float, default=0.998)
    args = parser.parse_args()

    main(args)