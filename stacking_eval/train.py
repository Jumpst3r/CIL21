import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from dataset import ArealDataset
from models.unet import StackedUNet

pl.seed_everything(42)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

import gc
from pprint import pprint

from sklearn.model_selection import KFold

from config import args

if __name__ == '__main__':

    pprint(vars(args))

    dataset = ArealDataset(root_dir_images='training/training/images/',
                           root_dir_gt='training/training/groundtruth/',
                           target_size=(args.res, args.res))
    kf = KFold(n_splits=2, shuffle=True)
    val_IoU = []
    val_F1 = []
    val_acc = []

    for train_indices, test_indices in kf.split(dataset):
        train_dataset = torch.utils.data.dataset.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.dataset.Subset(dataset, test_indices)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      pin_memory=True,
                                      num_workers=1)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.batch_size,
                                     pin_memory=True,
                                     num_workers=1)

        model = StackedUNet(lr=args.lr,
                            nb_blocks=args.nb_blocks,
                            unet_mode=args.unet_mode,
                            stacking_mode=args.stacking_mode,
                            loss_mode=args.loss_mode)
        trainer = pl.Trainer(max_epochs=args.max_epochs,
                             gpus=1,
                             stochastic_weight_avg=True,
                             precision=16,
                             deterministic=True,
                             checkpoint_callback=False,
                             progress_bar_refresh_rate=0,
                             logger=False)

        train_dataset.dataset.applyTransforms = True
        trainer.fit(model, train_dataloader)

        test_dataset.dataset.applyTransforms = False
        results = trainer.test(model, test_dataloader, verbose=False)[0]

        val_IoU.append(results['results'][0])
        val_F1.append(results['results'][1])
        val_acc.append(results['results'][2])
        print('IoU = ', results['results'][0])
        print('F1 = ', results['results'][1])
        print('acc = ', results['results'][2])

        del model
        del trainer
        del train_dataset
        del test_dataset
        del train_dataloader
        del test_dataloader
        gc.collect()
        torch.cuda.empty_cache()

    print('mean/std IoU:', np.array(val_IoU).mean(), np.array(val_IoU).std())
    print('mean/std F1:', np.array(val_F1).mean(), np.array(val_F1).std())
    print('mean/std acc:', np.array(val_acc).mean(), np.array(val_acc).std())

    if args.ckpt_dir != '':
        # train on full dataset to get kaggle score:
        checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir)
        train_dataloader = DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      pin_memory=False,
                                      num_workers=8)
        model = StackedUNet(lr=args.lr,
                            nb_blocks=args.nb_blocks,
                            unet_mode=args.unet_mode,
                            stacking_mode=args.stacking_mode,
                            loss_mode=args.loss_mode)
        trainer = pl.Trainer(max_epochs=args.max_epochs,
                             gpus=1,
                             stochastic_weight_avg=True,
                             precision=16,
                             deterministic=True,
                             checkpoint_callback=checkpoint_callback)
        dataset.applyTransforms = True
        trainer.fit(model, train_dataloader)
