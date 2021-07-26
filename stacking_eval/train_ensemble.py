import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dataset import ArealDataset
from models.unet import StackedUNet
from pprint import pprint
import os.path as osp
from config import args
import glob

if __name__ == '__main__':
    pl.seed_everything(args.seed)

    pprint(vars(args))

    dataset = ArealDataset(root_dir_images='training/images/',
                           root_dir_gt='training/groundtruth/',
                           target_size=(args.res, args.res))

    # train on full dataset to get kaggle score:
    checkpoint_callback = ModelCheckpoint(dirpath=osp.join('ensemble_deepglobe', args.ckpt_dir))
    train_dataloader = DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  pin_memory=False,
                                  num_workers=4)
    model = StackedUNet(lr=args.lr,
                        nb_blocks=args.nb_blocks,
                        unet_mode=args.unet_mode,
                        stacking_mode=args.stacking_mode,
                        loss_mode=args.loss_mode,
                        use_scheduler=False)
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         gpus=1,
                         stochastic_weight_avg=True,
                         precision=16,
                         deterministic=True,
                         checkpoint_callback=checkpoint_callback)

    dataset.applyTransforms = True
    trainer.fit(model, train_dataloader)
