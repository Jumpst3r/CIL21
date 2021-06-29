
# Iter-UNET: Iterative Road Segmentation from Aerial Images

![header](header.png)

TODO: add (link to) final report pdf

## Installation

For convenience we included a `requirements.txt` file which can be used to install the same package we used as follows:

- Install the `python3.7` and `python3-pip` packages
- Install the virtualenv package: `pip3 install virtualenv`
- Create a python virtual environment (or skip this and install packages system-wide) : `virtualenv env`
- Activate the created environment: `source env/bin/activate`
- Install the requirements: `pip3 install -r requirements.txt`


## Reproduce Submission
The final submission can be repoduced with the following commands, run from the `submission` directory.

TODO: Nicolas

## Reproduce Baselines

The baselines (table ??) can be reproduced with the following commands, run from the `baselines` directory.
Outputs called `model_name_{f1, iou}.npy` are saved within the same directory and contain the metrics as a `folds x epochs` matrix.

```
bsub -W 240 -oo /your_path/fcn_50_gpu.txt -R "rusage[ngpus_excl_p=1, scratch=10000,mem=10000]" python ./training.py fcn_resnet50 eval 
bsub -W 240 -oo /your_path/fcn_101_gpu.txt -R "rusage[ngpus_excl_p=1, scratch=10000,mem=10000]" python ./training.py fcn_resnet101 eval  
bsub -W 240 -oo /your_path/dlv3_50_gpu.txt -R "rusage[ngpus_excl_p=1, scratch=10000,mem=10000]" python ./training.py deeplabv3_resnet50 eval 
bsub -W 240 -oo /your_path/dlv3_101_gpu.txt -R "rusage[ngpus_excl_p=1, scratch=10000,mem=10000]" python ./training.py deeplabv3_resnet101 eval 
```

## Reproduce Post-processing  

The post-processing (table ??) can be reproduced with the following commands, run from the `postp_pipeline` directory.
The output called `post_processing_performance.npy` is saved within the same directory and contains the result matrix.

```
cd ./postp_pipeline && mkdir infer_basic infer_test_augment infer_adaptive infer_thresh infer_crf 
bsub -W 240 -oo /your_path/all_models_pp_train_gpu.txt -R "rusage[ngpus_excl_p=1, scratch=10000,mem=15000]" python test_pipeline.py train opt_train 
bsub -W 240 -oo /your_path/all_models_pp_test_gpu.txt -R "rusage[ngpus_excl_p=1, scratch=10000,mem=15000]" python test_pipeline.py test opt_test
```

## Reproduce Refinement Experiments

The refinement evaluation experiments (table ??) can be repoduced with the following commands, run from the `stacking_eval` directory.

### UNet
#### 1 block
```
bsub -W 4:00 -o c_1 -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 1 --unet_mode classic --stacking_mode hourglass --loss_mode sum --ckpt_dir c_1 --max_epochs 150 --res 128
```

#### hourglass stacking
```
bsub -W 4:00 -o c_2 -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 2 --unet_mode classic --stacking_mode hourglass --loss_mode sum --ckpt_dir c_2 --max_epochs 150 --res 128
bsub -W 4:00 -o c_4 -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 4 --unet_mode classic --stacking_mode hourglass --loss_mode sum --ckpt_dir c_4 --max_epochs 150 --res 128
bsub -W 4:00 -o c_6 -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 6 --unet_mode classic --stacking_mode hourglass --loss_mode sum --ckpt_dir c_6 --max_epochs 150 --res 128
```

#### simple stacking
```
bsub -W 4:00 -o c_s_2 -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 2 --unet_mode classic --stacking_mode simple --loss_mode sum --ckpt_dir c_s_2 --max_epochs 150 --res 128
bsub -W 4:00 -o c_s_4 -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 4 --unet_mode classic --stacking_mode simple --loss_mode sum --ckpt_dir c_s_4 --max_epochs 150 --res 128
bsub -W 4:00 -o c_s_6 -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 6 --unet_mode classic --stacking_mode simple --loss_mode sum --ckpt_dir c_s_6 --max_epochs 150 --res 128
```

### UNet-ResNet

#### 1 block
```
bsub -W 4:00 -o cb_1 -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 1 --unet_mode classic-backbone --stacking_mode hourglass --loss_mode sum --ckpt_dir cb_1 --max_epochs 150 --res 128
```

#### hourglass stacking
```
bsub -W 4:00 -o cb_2 -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 2 --unet_mode classic-backbone --stacking_mode hourglass --loss_mode sum --ckpt_dir cb_2 --max_epochs 150 --res 128
bsub -W 4:00 -o cb_4 -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 4 --unet_mode classic-backbone --stacking_mode hourglass --loss_mode sum --ckpt_dir cb_4 --max_epochs 150 --res 128
bsub -W 4:00 -o cb_6 -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 6 --unet_mode classic-backbone --stacking_mode hourglass --loss_mode sum --ckpt_dir cb_6 --max_epochs 150 --res 128
```

#### simple stacking
```
bsub -W 4:00 -o cb_s_2 -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 2 --unet_mode classic-backbone --stacking_mode simple --loss_mode sum --ckpt_dir cb_s_2 --max_epochs 150 --res 128
bsub -W 4:00 -o cb_s_4 -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 4 --unet_mode classic-backbone --stacking_mode simple --loss_mode sum --ckpt_dir cb_s_4 --max_epochs 150 --res 128
bsub -W 4:00 -o cb_s_6 -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 6 --unet_mode classic-backbone --stacking_mode simple --loss_mode sum --ckpt_dir cb_s_6 --max_epochs 150 --res 128
```

### UNet-ResNet, with a single loss at the end
```
bsub -W 4:00 -o cb_6_last -R "rusage[ngpus_excl_p=1,mem=8096]" python train.py --nb_blocks 6 --unet_mode classic-backbone --stacking_mode hourglass --loss_mode last --ckpt_dir cb_6_last --max_epochs 150 --res 128
```