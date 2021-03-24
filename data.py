
def dataset_transform():
    images_names = sorted(glob.glob('/home/nicolas/new-dataset/test/sat/*.png'))
    gt_name = sorted(glob.glob('/home/nicolas/new-dataset/test/map/*.png'))

    for _ in range(50):
        for im_n, gt_n in zip(images_names, gt_name):
            print(im_n, gt_n)
            pil_im = Image.open(im_n)
            pil_gt = Image.open(gt_n)
            image =  transforms.ToTensor()(pil_im)
            gt = torch.tensor(np.where(np.array(pil_gt) >=  1, 1., 0.)).unsqueeze(0)
            transform = transforms.RandomResizedCrop(400, scale=(0.02, 0.02), ratio=(1.,1.))
            state = torch.get_rng_state()
            image = transform(image)
            torch.set_rng_state(state)
            gt = transform(gt)
            if gt.mean() < 0.02: continue
            gt = np.array(gt*255, dtype=np.uint8)[0]
            im = Image.fromarray(np.array(gt, dtype=np.uint8))
            global cnt
            im.save(f'training/training/groundtruth/custom-{cnt}-mask.png')
            npim = np.moveaxis(np.array(image*255, dtype=np.uint8), 0, -1)
            im = Image.fromarray(npim)
            im.save(f'training/training/images/custom-{cnt}-input.png')
            cnt += 1

