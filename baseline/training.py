import argparse
import time

import numpy as np
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
import getpass


from baseline_models import *
from dataset import ArealDataset

pl.seed_everything(42)
torch.backends.cudnn.benchmark = False


import gc

from sklearn.model_selection import KFold

if __name__ == '__main__':
    USERNAME = getpass.getuser()
    dataset = ArealDataset(root_dir_images='training/images/',
                           root_dir_gt='training/groundtruth/',
                           target_size=(128, 128))
    cross_val = 2
    kf = KFold(n_splits=cross_val, shuffle=True)

    epochs = 300

    val_IoU = []
    val_F1 = []
    val_acc = []

    idx = np.random.permutation(np.arange(100))

    parser = argparse.ArgumentParser()
    parser.add_argument("key", help="model which should be trained", type=str)
    parser.add_argument(
        "mode",
        help=
        "enter \"eval\" or \"train\" to eval with CV or train on whole set for kaggle sub",
        type=str)
    key = vars(parser.parse_args())['key']
    mode = vars(parser.parse_args())['mode']

    def get_trainer():
        if torch.cuda.is_available():
            return pl.Trainer(max_epochs=epochs,
                              gpus=1,
                              deterministic=True,
                              progress_bar_refresh_rate=0,
                              logger=False)  #
        else:
            return pl.Trainer(max_epochs=epochs,
                              deterministic=True,
                              progress_bar_refresh_rate=0,
                              logger=False)

    #for key in seg_models.keys():
    def eval(key):
        print("training: ", key)
        fold = 0
        iou = np.zeros((cross_val, epochs))
        f1 = np.zeros((cross_val, epochs))
        acc = np.zeros((cross_val, epochs))
        for train_indices_plain, test_indices_plain in kf.split(dataset):
            train_indices = [idx[i] for i in train_indices_plain]
            test_indices = [idx[i] for i in test_indices_plain]
            train_dataset = torch.utils.data.dataset.Subset(
                dataset, train_indices)
            test_dataset = torch.utils.data.dataset.Subset(
                dataset, test_indices)
            train_dataset.dataset.applyTransforms = True
            train_dataloader = DataLoader(train_dataset,
                                          batch_size=5,
                                          pin_memory=True,
                                          num_workers=8)
            train_dataloader.dataset.applyTransforms = True
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=5,
                                         pin_memory=True,
                                         num_workers=1)
            test_dataloader.dataset.applyTransforms = False

            model = VisionBaseline(seg_models[key], model_opts[key], loss[key],
                                   optimizer[key], optimizer_options[key],
                                   epochs)
            trainer = get_trainer()

            start = time.time()
            trainer.fit(model, train_dataloader, test_dataloader)
            end = time.time()


            #0th entry is sanity check, drop that
            iou[fold, :] = model.val_iou[1:-1]
            f1[fold, :] = model.val_f1[1:-1]
            acc[fold, :] = model.val_acc[1:-1]

            fold += 1
            del model
            del trainer
            del train_dataset
            del test_dataset
            del train_dataloader
            del test_dataloader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print('IoU: ', iou.mean(), iou.std())
        print('f1: ', f1.mean(), f1.std())
        print('acc: ', acc.mean(), acc.std())

    def train(key):
        print("training for kaggle eval: ", key)
        train_dataset = torch.utils.data.dataset.Subset(
            dataset, [i for i in range(100)])
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=5,
                                      pin_memory=True,
                                      num_workers=8)
        model = VisionBaseline(seg_models[key], model_opts[key], loss[key],
                               optimizer[key], optimizer_options[key], epochs)
        trainer = get_trainer()

        start = time.time()
        trainer.fit(model, train_dataloader)
        end = time.time()
        print("time elapsed: ", end - start, "s epochs: ", epochs)
        folder = f'/cluster/scratch/{USERNAME}/CIL21/'
        PATH = folder + key + "_trained.pt"
        torch.save(model.state_dict(), PATH)

    if mode == "eval":
        eval(key)
    if mode == "train":
        train(key)
