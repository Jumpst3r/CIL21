#adapt search path for imports
import sys
sys.path.append('../baseline')
sys.path.append('../submission')
from dataset import ArealDataset, ArealDatasetIdx
#from utils import IoU, F1
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
from models.unet import StackedUNet


def F1(inputs, targets):
    inputs = torch.round(inputs)
    TP = (inputs * targets).sum()
    FP = ((1 - targets) * inputs).sum()
    FN = (targets * (1 - inputs)).sum()

    return TP / (TP + 0.5 * (FP + FN))


def IoU(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # import pdb; pdb.set_trace()
    y_true = y_true.type(torch.int32)
    y_pred = torch.round(y_pred).type(torch.int32)
    y_pred = (y_pred == 1)
    y_true = (y_true == 1)
    eps = 1e-4
    intersection = (y_pred & y_true).float().sum((1, 2))
    union = (y_pred | y_true).float().sum((1, 2))

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def get_trainer(epochs):
    if torch.cuda.is_available():
        return pl.Trainer(max_epochs=epochs, gpus=1, deterministic=True, progress_bar_refresh_rate=4, logger=True)  #
    else:
        return pl.Trainer(max_epochs=epochs, deterministic=True, progress_bar_refresh_rate=4, logger=True)


def train_submission(opts):
    dataset = ArealDataset(root_dir_images=root_dir_images, root_dir_gt=root_dir_gt,
                           target_size=(opts['target_res'], opts['target_res']), applyTransforms=opts['augment'])

    for i, (test, train) in enumerate(opts['folds']):
        if(i != 0):
            continue
        train_dataset = torch.utils.data.dataset.Subset(dataset, train)
        test_dataset = torch.utils.data.dataset.Subset(dataset, test)
        test_dataset.applyTransforms = opts['augment']
        train_dataset.applyTransforms = opts['augment']
        train_dataloader = DataLoader(train_dataset, batch_size=opts['batch_size'], pin_memory=True, num_workers=8)
        #test_dataloader = DataLoader(test_dataset, batch_size=opts['batch_size'], pin_memory=True, num_workers=8)

        model = StackedUNet()
        trainer = get_trainer(opts['epochs'])
    
        start = time.time()
        trainer.fit(model, train_dataloader)
        end = time.time()
        save_path = model_dir + '/Unet_' + str(i) + model_suffix
        torch.save(model.state_dict(), save_path)
        print("fold:", i,  "time: ", end - start)
        del model
        torch.cuda.empty_cache()


def evaluate(img, lbl):
    return np.array(F1(img, lbl).detach().cpu().numpy()), np.array(IoU(img, lbl).detach().cpu().numpy())


def full_img_nr(idx):
    if idx < 10:
        return '00' + str(idx)
    if idx >= 10 and idx < 100:
        return '0' + str(idx)
    return str(idx)


def infer_test_augment(dataset, model):
    basic_dir = './infer_test_augment'
    i = 0
    iou_ls = []
    f1_ls = []
    iou_ls_r = []
    f1_ls_r = []

    for batch, idx in dataset:
        i += 1
        img, lbl = batch
        img0 = img.unsqueeze(0)
        y0 = torch.sigmoid(model(img0).squeeze(0))
        y1 = torch.sigmoid(torch.flip(model(torch.flip(img0, [2])), [2]).squeeze(0))
        y2 = torch.sigmoid(torch.flip(model(torch.flip(img0, [3])), [3]).squeeze(0))
        y = torch.cat((y0, y1, y2))
        for j in range(1, 4):
            imgr = torch.rot90(img0, j, [2, 3])
            imgrr = torch.sigmoid(model(imgr).squeeze(0))
            imgrf = torch.sigmoid(torch.flip(model(torch.flip(imgr, [2])), [2]).squeeze(0))
            imgrh = torch.sigmoid(torch.flip(model(torch.flip(imgr, [3])), [3]).squeeze(0))
            imgrr = torch.rot90(imgrr, 4 - j, [1, 2])
            imgrf = torch.rot90(imgrf, 4 - j, [1, 2])
            imgrh = torch.rot90(imgrh, 4 - j, [1, 2]) 
            y = torch.cat((y, imgrr, imgrf, imgrh))
        y_tens = torch.mean(y, 0).unsqueeze(0)
        imout = y_tens.squeeze(0)
        imout = np.array(imout.detach().cpu().numpy(), dtype=np.float32)

        f1, iou = evaluate(y_tens, lbl)
        f1_r, iou_r = evaluate(y0, lbl)
        iou_ls.append(iou)
        f1_ls.append(f1)
        iou_ls_r.append(iou_r)
        f1_ls_r.append(f1_r)
        print(i, iou, f1, idx + 1, "|", f1-f1_r, iou-iou_r)
        
        im = Image.fromarray(np.array(imout * 255, dtype=np.uint8))
        fname = '/satImage_' + full_img_nr(idx+1) + '.png'
        im.save(basic_dir+fname)
    print("infer augment: iou: ", np.mean(iou_ls), "f1: ", np.mean(f1_ls), "iou ref: ", np.mean(iou_ls_r), "f1_ref: ", np.mean(f1_ls_r), "diff iou: ", np.mean(iou_ls) - np.mean(iou_ls_r), "diff f1: ", np.mean(f1_ls) - np.mean(f1_ls_r))
    return iou_ls, f1_ls


def infer_basic(dataset, model):
    # infer: iou:  0.7317977 f1: 0.82485026
    basic_dir = './infer_basic'
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

        f1, iou = evaluate(y_tens, lbl)
        iou_ls.append(iou)
        f1_ls.append(f1)
        print(i, iou, f1, idx + 1)

        im = Image.fromarray(np.array(imout * 255, dtype=np.uint8))
        fname = '/satImage_' + full_img_nr(idx+1) + '.png'
        im.save(basic_dir+fname)
    print("infer: iou: ", np.mean(iou_ls), "f1: ", np.mean(f1_ls))
    return iou_ls, f1_ls

def crf(dataset, lbl_path):
    # best: iou:  0.73279184 f1:  0.827103
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
        y = torch.tensor(out, dtype=torch.float32).unsqueeze(0)

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


def test(opts):
    iou_basic = []
    f1_basic = []
    iou_augment = []
    f1_augment = []
    for i, (test, train) in enumerate(opts['folds']):
        if i != 3:
            continue
        path = model_dir + '/Unet_' + str(i) + model_suffix
        model = StackedUNet()
        model.load_state_dict(torch.load(path))
        model.eval()

        dataset = ArealDatasetIdx(root_dir_images=root_dir_images, root_dir_gt=root_dir_gt,
                               target_size=(opts['target_res'], opts['target_res']))
        test_dataset = torch.utils.data.dataset.Subset(dataset, test)
        ious_basic, f1s_basic = infer_basic(test_dataset, model)
        iou_basic += ious_basic
        f1_basic += f1s_basic

        ious_augment, f1s_augment = infer_test_augment(dataset, model)
        iou_augment += ious_augment
        f1_augment += f1s_augment
    print("basic inference, iou: ", np.mean(iou_basic), "f1: ", np.mean(f1_basic))
    print("augment inference, iou: ", np.mean(iou_augment), "f1: ", np.mean(f1_augment))



pl.seed_everything(2)

idx = np.random.permutation(np.arange(100))
folds = [(idx[0:25], idx[25:100]), (idx[25:50], np.concatenate((idx[0:25], idx[50:100]))), (idx[50:75], np.concatenate((idx[0:50], idx[75:100]))), (idx[75:100], idx[0:75])]

opt_train = {'target_res': 128, 'batch_size': 5, 'epochs': 200, 'augment': True, 'folds': folds}
opt_test = {**opt_train, 'infer': infer_test_augment, 'pp': crf}
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
    mode = vars(parser.parse_args())['mode']
    opts = vars(parser.parse_args())['opts']

    if mode == 'train':
        train_submission(options[opts])

    if mode == 'test':
        test(options[opts])
