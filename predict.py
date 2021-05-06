
import torch
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
import os
import cv2

from data.dataset_test import ArealDatasetTest
from models.stacked_unet_plwrapper import StackedUNetPL

os.makedirs('out', exist_ok=True)

means = [0.33415824789915965, 0.3313150847338936, 0.2974799467787115]
stds = [0.18386573251831514, 0.1774855852173398, 0.17591074700521442]

weight_path = 'weights/StackedUNet_bs-1_ns-4_ac-8_lr-0.0001_bil-False_is-256_p-16_n-instance.ckpt'
model = StackedUNetPL.load_from_checkpoint(weight_path).cuda()

torch.set_grad_enabled(False)
model.eval()

testds = ArealDatasetTest('test_images/test_images/', (608,608), means, stds)
test_dataloader = DataLoader(testds, batch_size=1, shuffle=False, num_workers=0)

for img, name in iter(test_dataloader):
    img = img.cuda()

    logits_list = model(img)
    logits = logits_list[-1]
    prob = torch.sigmoid(logits)
    thresh = prob.clone()
    thresh[prob > 0.5] = 1
    thresh[thresh < 1] = 0

    for j, n in enumerate(name):
        final_pred = (thresh[j][0].detach().cpu().numpy() * 255).astype(np.uint8)
        final_pred = cv2.resize(final_pred, (608,608), interpolation=cv2.INTER_NEAREST)

        im_pil = Image.fromarray(final_pred)

        im_pil.save('out/' + n)
