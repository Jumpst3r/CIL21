
# Iter-UNET: Iterative Road Segmentation from Aerial Images


![header](header.png)

This repository contains the code & report for the CIL21 road segmentation project.

Our team name on kaggle is `PixelSurfers`, an our team consists of:

 - Nicolas Dutly
 - Janik Lobsiger
 - Fabijan Dokic

The focus of our project is the investigation of if and how stacking & combining UNet models can lead to performance benefits.

We call our architecture IterUNet. This code repository is structured as follows:

 - `submission`: Contains the code that was used to create the end submission. Also applies ensembling and test time augmentation (not applied on other experiments)
 - `baseline`: Contains the code needed to reproduce the FCN-ResNet and deeplab baseline results
 - `stacking_eval`: Contains the code needed to reproduce our model evaluation experiments

 The files in each directory are somewhat similar / redundant but we grouped them into different directories for clarity.

### Evaluation

To evaluate the impact of different variants of our model and compare them to the benchmarks, we always do 2-fold cross validation and report the following metrics:

 - Intersection over Union (IoU)
 - Pixel Accuracy

We chose not to include the kaggle score for comparative experiments as the limited public test-set does not allow for significant comparisons to be made.

Of course, for the final submission (which includes test time augmentation and ensembling), we do report the (public) kaggle score.

### Installation

For convenience we included a `requirements.txt` file which can be used to install the same package we used as follows:

- Create a python environment with the tool of your choice (`pip-env`, `virtualenv` etc)
- Activate the created environment: `source env/bin/activate`
- Install the requirements: `pip3 install -r requirements.txt`


### Reproduce Submission
The final submission can be repoduced with the following commands, run from the `submission` directory.

TODO: Nicolas

### Reproduce Baselines

The FCN-ResNet & Deeplabv3 baselines can be reproduced with the following commands, run from the `baselines` directory.

```
bsub -W 240 -oo fcn_101_gpu.txt -R "rusage[ngpus_excl_p=1, scratch=10000,mem=10000]" -R "select[gpu_model0==GeForceRTX2080Ti]" python ./training.py fcn_resnet101 eval

bsub -W 240 -oo deeplab_101_gpu.txt -R "rusage[ngpus_excl_p=1, scratch=10000,mem=10000]" -R "select[gpu_model0==GeForceRTX2080Ti]" python ./training.py deeplabv3_resnet101 eval

```

The standart UNet baseline can be reproduced by navigating to the `stacking_eval` directory and running:

```
bsub -W 4:00 -o unet_1 -R "rusage[ngpus_excl_p=1,mem=8096]" -R "select[gpu_model0==GeForceRTX2080Ti]" python train.py --nb_blocks 1 --unet_mode classic --stacking_mode hourglass --loss_mode sum --max_epochs 300 --res 128
```

## Reproduce IterUNet Experiments

Our model evaluation experiments can be repoduced with the following commands, run from the `stacking_eval` directory.


#### hourglass stacking

```
bsub -W 4:00 -o unet_hg_2 -R "rusage[ngpus_excl_p=1,mem=8096]" -R "select[gpu_model0==GeForceRTX2080Ti]" python train.py --nb_blocks 2 --unet_mode classic --stacking_mode hourglass --loss_mode sum --max_epochs 300 --res 128
bsub -W 4:00 -o unet_hg_4 -R "rusage[ngpus_excl_p=1,mem=8096]" -R "select[gpu_model0==GeForceRTX2080Ti]" python train.py --nb_blocks 4 --unet_mode classic --stacking_mode hourglass --loss_mode sum --max_epochs 300 --res 128
```

#### simple stacking
```
bsub -W 4:00 -o unet_simple_2 -R "rusage[ngpus_excl_p=1,mem=8096]" -R "select[gpu_model0==GeForceRTX2080Ti]" python train.py --nb_blocks 2 --unet_mode classic --stacking_mode simple --loss_mode sum --max_epochs 300 --res 128
bsub -W 4:00 -o unet_simple_4 -R "rusage[ngpus_excl_p=1,mem=8096]" -R "select[gpu_model0==GeForceRTX2080Ti]" python train.py --nb_blocks 4 --unet_mode classic --stacking_mode simple --loss_mode sum --max_epochs 300 --res 128
```

### UNet-ResNet


#### hourglass stacking
```
bsub -W 4:00 -o resunet_hg_2 -R "rusage[ngpus_excl_p=1,mem=8096]" -R "select[gpu_model0==GeForceRTX2080Ti]" python train.py --nb_blocks 2 --unet_mode classic-backbone --stacking_mode hourglass --loss_mode sum --max_epochs 300 --res 128
bsub -W 4:00 -o resunet_hg_4 -R "rusage[ngpus_excl_p=1,mem=8096]" -R "select[gpu_model0==GeForceRTX2080Ti]" python train.py --nb_blocks 4 --unet_mode classic-backbone --stacking_mode hourglass --loss_mode sum --max_epochs 300 --res 128
```

#### simple stacking
```
bsub -W 4:00 -o resunet_simple_2 -R "rusage[ngpus_excl_p=1,mem=8096]" -R "select[gpu_model0==GeForceRTX2080Ti]" python train.py --nb_blocks 2 --unet_mode classic-backbone --stacking_mode simple --loss_mode sum --max_epochs 300 --res 128
bsub -W 4:00 -o resunet_simple_4 -R "rusage[ngpus_excl_p=1,mem=8096]" -R "select[gpu_model0==GeForceRTX2080Ti]" python train.py --nb_blocks 4 --unet_mode classic-backbone --stacking_mode simple --loss_mode sum --max_epochs 300 --res 128
```
