#adapt search path for imports
import sys
sys.path.append('../baseline')
from dataset import ArealDataset, ArealDatasetIdx
from utils import IoU, F1
from baseline_models import *
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101
import argparse
import time
import pytorch_lightning as pl
from sklearn.model_selection import KFold
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn.functional as F
from PIL import Image
from crf import dense_crf
import cv2 as cv


def get_trainer(epochs):
    if torch.cuda.is_available():
        return pl.Trainer(max_epochs=epochs, gpus=1, deterministic=True, progress_bar_refresh_rate=0, logger=False)  #
    else:
        return pl.Trainer(max_epochs=epochs, deterministic=True, progress_bar_refresh_rate=0, logger=False)


def train(opts):
    dataset = ArealDataset(root_dir_images=root_dir_images, root_dir_gt=root_dir_gt,
                           target_size=(opts['target_res'], opts['target_res']))
    train_dataset = torch.utils.data.dataset.Subset(dataset, opts['train_idx'])
    test_dataset = torch.utils.data.dataset.Subset(dataset, opts['test_idx'])
    test_dataset.applyTransforms = opts['augment']
    train_dataset.applyTransforms = opts['augment']
    train_dataloader = DataLoader(train_dataset, batch_size=opts['batch_size'], pin_memory=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=opts['batch_size'], pin_memory=True, num_workers=8)

    for key in seg_models.keys():
        model = VisionBaseline(seg_models[key], model_opts[key], loss[key], optimizer[key], optimizer_options[key], opts['epochs'])
        trainer = get_trainer(opts['epochs'])

        start = time.time()
        trainer.fit(model, train_dataloader, test_dataloader)
        end = time.time()
        print("model: ", key, " time: ", end - start, "s iou: ", model.val_iou, " f1: ", model.val_f1)
        save_path = model_dir + '/' + key + model_suffix
        torch.save(model.state_dict(), save_path)


def evaluate(img, lbl):
    return np.array(F1(img, lbl).detach().cpu().numpy()), np.array(IoU(img, lbl).detach().cpu().numpy())


def full_img_nr(idx):
    if idx < 10:
        return '00' + str(idx)
    if idx >= 10 and idx < 100:
        return '0' + str(idx)
    return str(idx)


def infer_basic(dataset, model):
    # infer: iou:  0.7317977 f1: 0.82485026 dlv3_101@256
    basic_dir = '/infer_basic'
    i = 0
    iou_ls = []
    f1_ls = []
    for batch, idx in dataset:
        i += 1
        img, lbl = batch
        img = img.unsqueeze(0)
        y = model(img)
        y_tens = torch.sigmoid(y[0])
        imout = np.array(y_tens.detach().cpu().numpy(), dtype=np.float32)
        imout = imout.squeeze(0)

        f1, iou = evaluate(y, lbl)
        iou_ls.append(iou)
        f1_ls.append(f1)
        print(i, iou, f1, idx + 1)

        im = Image.fromarray(np.array(imout * 255, dtype=np.uint8)) #.resize((orig_res, orig_res))
        fname = '/satImage_' + full_img_nr(idx+1) + '.png'
        dir = base_dir + basic_dir + fname
        #im.save(dir)
        dir = '.' + basic_dir
        im.save(dir+fname)
    print("infer: iou: ", np.mean(iou_ls), "f1: ", np.mean(f1_ls))
    return dir

def crf(dataset, lbl_path):
    # best: iou:  0.73279184 f1:  0.827103, dlv3_101@256
    i = 0
    iou_ls = []
    f1_ls = []
    basic_dir = '/infer_crf'
    for batch, idx in dataset:
        i += 1
        img, lbl = batch
        pred_path = lbl_path + '/satImage_' + full_img_nr(idx+1) + '.png'
        pred = np.array(Image.open(pred_path))
        n = 1 / 255
        pred = np.array(pred * n, dtype=np.float32)

        bilat = np.array(img.squeeze(0).detach().cpu().numpy() * 255, dtype=np.uint8)
        bilat = np.reshape(bilat, (256, 256, 3))

        out = dense_crf(bilat, pred)
        y = torch.tensor(out).unsqueeze(0)

        f1, iou = evaluate(y, lbl)
        iou_ls.append(iou)
        f1_ls.append(f1)
        print(i, iou, f1, idx + 1)

        im = Image.fromarray(np.array(out * 255, dtype=np.uint8)) #.resize((orig_res, orig_res))
        fname = '/satImage_' + full_img_nr(idx+1) + '.png'
        dir = '.' + basic_dir
        im.save(dir+fname)
    print("crf: iou: ", np.mean(iou_ls), "f1: ", np.mean(f1_ls))


def thresh(dataset, lbl_path):
    # best: same as infer_basic!, with thresh = 0.5
    i = 0
    iou_ls = []
    f1_ls = []
    basic_dir = '/infer_thresh'
    thresh = 0.5
    for batch, idx in dataset:
        i += 1
        _, lbl = batch
        pred_path = lbl_path + '/satImage_' + full_img_nr(idx+1) + '.png'
        pred = np.array(Image.open(pred_path))
        n = 1 / 255
        pred = np.array(pred * n, dtype=np.float32)

        out = np.array(pred > thresh)
        y = torch.tensor(out).unsqueeze(0)

        f1, iou = evaluate(y, lbl)
        iou_ls.append(iou)
        f1_ls.append(f1)
        print(i, iou, f1, idx + 1)

        im = Image.fromarray(np.array(out * 255, dtype=np.uint8)) #.resize((orig_res, orig_res))
        fname = '/satImage_' + full_img_nr(idx+1) + '.png'
        dir = '.' + basic_dir
        im.save(dir+fname)
    print("thresh: iou: ", np.mean(iou_ls), "f1: ", np.mean(f1_ls))

def adaptive(dataset, lbl_path):
    # best: adaptive: iou:  0.7309903 f1:  0.8259104964763566
    i = 0
    iou_ls = []
    f1_ls = []
    basic_dir = '/infer_adaptive'
    thresh = 0.5
    for batch, idx in dataset:
        i += 1
        img, lbl = batch
        pred_path = lbl_path + '/satImage_' + full_img_nr(idx+1) + '.png'
        pred = np.array(Image.open(pred_path))
        #pred = np.array(pred > 127, dtype=np.uint8)
        
        out = cv.GaussianBlur(pred, (9,9), 0)
        _, out = cv.threshold(out, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        #out = cv.adaptiveThreshold(pred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 1555, 0)
        #out = cv.adaptiveThreshold(out, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 1555, 0)
        #out = cv.adaptiveThreshold(out, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 1555, 0) 
        #out = cv.adaptiveThreshold(out, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 1555, 0) 
        #out = cv.adaptiveThreshold(out, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 1555, 0) 
        out = out / 255
        
        y = torch.tensor(out).unsqueeze(0)

        f1, iou = evaluate(y, lbl)
        iou_ls.append(iou)
        f1_ls.append(f1)
        print(i, iou, f1, idx + 1)

        im = Image.fromarray(np.array(out*255, dtype=np.uint8)) #.resize((orig_res, orig_res))
        fname = '/satImage_' + full_img_nr(idx+1) + '.png'
        dir = '.' + basic_dir
        im.save(dir+fname)
    print("adaptive: iou: ", np.mean(iou_ls), "f1: ", np.mean(f1_ls))


def test(key, opts):
    path = model_dir + '/' + key + model_suffix
    #path = './deeplabv3_resnet101_trained.pt'
    model = VisionBaseline(seg_models[key], base_model_options, F.binary_cross_entropy_with_logits, optimizer[key],
                           base_adam_options, opts['epochs'])
    model.load_state_dict(torch.load(path))
    model.eval()

    dataset = ArealDatasetIdx(root_dir_images=root_dir_images, root_dir_gt=root_dir_gt,
                           target_size=(opts['target_res'], opts['target_res']))
    test_dataset = torch.utils.data.dataset.Subset(dataset, opts['test_idx'])

    #infer_func = opts['infer']
    #infer_path = infer_func(test_dataset, model)
    infer_path = './infer_basic'
    pp_func = opts['pp']
    pp_path = pp_func(test_dataset, infer_path)

    #f1, iou = evaluate(pp_path)


pl.seed_everything(2)

idx = np.random.permutation(np.arange(100))
test_indices = [idx[i] for i in range(0, 25)]
train_indices = [idx[i] for i in range(25, 100)]

opt_train = {'target_res': 256, 'batch_size': 5, 'epochs': 50, 'augment': True, 'train_idx': train_indices, 'test_idx': test_indices}
opt_test = {**opt_train, 'infer': infer_basic, 'pp': adaptive}
options = {'opt_train': opt_train, 'opt_test': opt_test}

orig_res = 400

base_dir = '/cluster/scratch/fdokic/CIL21'
model_dir = base_dir + '/pp_models'
model_suffix = "_cv_trained.pt"

root_dir_gt = '../baseline/training/groundtruth/'
root_dir_images = '../baseline/training/images/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="enter \"test\" or \"train\" to test pp pipeline on trained model or to train a model", type=str)
    parser.add_argument("opts", help="enter which options dict should be passed to method", type=str)
    parser.add_argument("key", help="path to model which should evaluated or to be trained", type=str)
    mode = vars(parser.parse_args())['mode']
    opts = vars(parser.parse_args())['opts']
    key = vars(parser.parse_args())['key']

    if mode == 'train':
        train(options[opts])

    if mode == 'test':
        test(key, options[opts])
