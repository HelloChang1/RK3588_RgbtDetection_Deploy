import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import random
import glob
import math
from tqdm import tqdm
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, dataloader
from utils.general import seed_worker,img2label_paths,xyxy2xywhn, LOGGER, TQDM_BAR_FORMAT,PIN_MEMORY,NUM_THREADS,TQDM_BAR_FORMAT,verify_image_label, \
    xywhn2xyxy,xyxy2xywhn,xyn2xy,get_hash,letterbox
from tqdm import tqdm

class RGBTDataloader(Dataset):
    # RGB-T train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    # rand_interp_methods 包含不同的插值方法，后续可能在图像缩放时使用。
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=1,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix=''):
        # 属性初始化
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = None

        #读取验证文件夹路径
        try:
            # 文件读取
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t]  # to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{prefix}{p} does not exist')
            
            # 根据文件名中的后缀，将图像路径分为 RGB 图像文件和 Thermal 图像文件，并将它们排序。此步骤使得图像能够被配对。
            # yuhang: Read RGB images and infrared images separately according to the suffix
            # self.rgb_files = sorted(x.replace('/', os.sep) for x in f if '_rgb' in x.split('.')[0]) 
            # self.t_files = sorted(x.replace('/', os.sep) for x in f if '_t' in x.split('.')[0]) 
            self.rgb_files = sorted(x.replace('/', os.sep) for x in f if x.endswith('_rgb.png')) 
            self.t_files = sorted(x.replace('/', os.sep) for x in f if x.endswith('_t.png')) 
            # 检查图像文件有效性
            # self.im_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.rgb_files, f'{prefix}No rgb images found'
            assert self.t_files, f'{prefix}No t images found'
            # yuhang: Check if the images are paired
            assert len(self.rgb_files) == len(self.t_files), f'{prefix}rgb images number is not equal t images {len(self.rgb_files)} != {len(self.t_files)}'
            for index in range(len(self.rgb_files)):
               assert os.path.basename(self.rgb_files[index]).split('_rgb')[0] == os.path.basename(self.t_files[index]).split('_t')[0], f'{prefix} index:{index}'
        
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\n') from e

        # 检查缓存和标签
        # Check cache
        self.label_files = img2label_paths(self.t_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.rgb_files + self.t_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # 缓存结果输出
        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists:
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels found in {cache_path}, can not start training. '
       
        # 读取缓存的标签和图像文件
        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))  # number of labels
        assert nl > 0 or not augment, f'{prefix}All labels empty in {cache_path}, can not start training. '
        self.labels = list(labels)
        self.shapes = np.array(shapes)

        # yuhang: Check if the images in cache are paired
        self.rgb_files = [x for x in cache.keys() if '_rgb' in x.split('/')[-1]]  # update
        self.t_files = [x for x in cache.keys() if '_t' in x.split('/')[-1]]
        self.label_files = img2label_paths([x for x in cache.keys() if '_t' in x.split('/')[-1]])  # update
        for index in range(len(self.rgb_files)):
            assert os.path.basename(self.rgb_files[index]).split('_rgb')[0] == os.path.basename(self.t_files[index]).split('_t')[0], f'{prefix} index:{index}'
            
        # Filter images
        # yuhang: Filter RGB and infrared images separately
        if min_items:
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            LOGGER.info(f'{prefix}{n - len(include)}/{n} images filtered from dataset')
            self.rgb_files = [self.rgb_files[i] for i in include]
            self.t_files = [self.t_files[i] for i in include]     
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]  # wh

        # Create indices
        
        # 计算图像的数量 n，以及根据批次大小计算批次索引 bi 和批次数量 nb。
        # 更新实例属性 self.batch, self.n, 和 self.indices。
        n = len(self.shapes)//2  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        # Rectangular Training
        # yuhang: Rectangular RGB and infrared images separately
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes[0:n]  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.rgb_files = [self.rgb_files[i] for i in irect]
            self.t_files = [self.t_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride
        # 图像缓存处理
        # Cache images into RAM/disk for faster training
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.rgb_files] + [Path(f).with_suffix('.npy') for f in self.t_files]

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{prefix}Scanning {path.parent / path.stem}...'

        # yuhang: Verify RGB and infrared images separately
        with Pool(NUM_THREADS) as pool:
            rgb_pbar = tqdm(pool.imap(verify_image_label, zip(self.rgb_files, self.label_files, repeat(prefix))),
                        desc=desc,
                        total=len(self.t_files),
                        bar_format=TQDM_BAR_FORMAT)
            for rgb_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in rgb_pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if rgb_file:
                    x[rgb_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                rgb_pbar.desc = f'{desc} {nf} rgb images, {nm + ne} backgrounds, {nc} corrupt'
        rgb_pbar.close()

        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        with Pool(NUM_THREADS) as pool:
            t_pbar = tqdm(pool.imap(verify_image_label, zip(self.t_files, self.label_files, repeat(prefix))),
                        desc=desc,
                        total=len(self.t_files),
                        bar_format=TQDM_BAR_FORMAT)
            for t_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in t_pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if t_file:
                    x[t_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                t_pbar.desc = f'{desc} {nf} t images, {nm + ne} backgrounds, {nc} corrupt'
        t_pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING ⚠️ No labels found in {path}. ')
        x['hash'] = get_hash(self.label_files + self.rgb_files + self.t_files)
        x['results'] = nf, nm, ne, nc, len(self.t_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def __len__(self):
        return len(self.t_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']

        # yuhang: In data augmentation, RGB and infrared images need to be enhanced in pairs
        # Load image
        img_rgb, _, _ = self.load_image(index, imtype='rgb')
        img_t, (h0, w0), (h, w) = self.load_image(index, imtype='t')


        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img_rgb, _, _ = letterbox(img_rgb, shape)
        img_t, ratio, pad = letterbox(img_t, shape)
        
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img_t.shape[1], h=img_t.shape[0], clip=True, eps=1E-3)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img_rgb = img_rgb.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_rgb = np.ascontiguousarray(img_rgb)
        img_t = img_t.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_t = np.ascontiguousarray(img_t)

        return torch.from_numpy(img_rgb), torch.from_numpy(img_t), labels_out, self.rgb_files[index], self.t_files[index], shapes


    def load_image(self, i, imtype='t'):
        # Loads 1 image from specify the type of dataset index 'i', returns (im, original hw, resized hw)
        # imtype mean the type of dataset, rgb/t
        if imtype == 't': 
            im, f, fn = self.ims[i], self.t_files[i], self.npy_files[i],
        elif imtype == 'rgb':
            im, f, fn = self.ims[i], self.rgb_files[i], self.npy_files[i],
        else :
            raise Exception(f'imtype only can be rgb/t!\n') 
        
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized


    @staticmethod
    def collate_fn(batch):
        im_rgb, im_t, label, rgb_path, t_path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im_rgb, 0), torch.stack(im_t, 0), torch.cat(label, 0), rgb_path, t_path, shapes

    @staticmethod
    def collate_fn4(batch):
        im_rgb, im_t, label, rgb_path, t_path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        im4_rgb, im4_t, label4, rgb_path4, t_path4, shapes4 = [], [], [], rgb_path[:n], t_path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im1 = F.interpolate(im_rgb[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear',
                                    align_corners=False)[0].type(im_rgb[i].type())
                im2 = F.interpolate(im_t[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear',
                                    align_corners=False)[0].type(im_t[i].type())
                lb = label[i]
            else:
                im1 = torch.cat((torch.cat((im_rgb[i], im_rgb[i + 1]), 1), torch.cat((im_rgb[i + 2], im_rgb[i + 3]), 1)), 2)
                im2 = torch.cat((torch.cat((im_t[i], im_t[i + 1]), 1), torch.cat((im_t[i + 2], im_t[i + 3]), 1)), 2)

                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4_rgb.append(im1)
            im4_t.append(im2)

            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4_rgb, 0), torch.stack(im4_rgb, 0), torch.cat(label4, 0), rgb_path4, t_path4, shapes4



class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


def create_rgbtdataloader(path,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False,
                      seed=0):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    dataset = RGBTDataloader(
        path,
        imgsz,
        batch_size,
        augment=augment,  # augmentation
        hyp=hyp,  # hyperparameters
        rect=rect,  # rectangular batches
        cache_images=cache,
        single_cls=single_cls,
        stride=int(stride),
        pad=pad,
        image_weights=image_weights,
        prefix=prefix)

    # 将 batch_size 设置为较小的值，以避免请求的批次大小超过数据集大小。
    batch_size = min(batch_size, len(dataset))
    nw = workers # number of workers
    # 如果不是在分布式模式下，则不使用采样器，将其设置为 None；否则，使用 DistributedSampler 来在分布式训练中处理数据的平衡和洗牌。
    sampler = None
    loader = InfiniteDataLoader  # only DataLoader allows for attribute updates
    # 创建一个 PyTorch 随机数生成器并设置种子值，以确保可重复性。种子值包括一些固定的常量、传入的 seed 和进程 RANK。
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed)
    # 返回数据加载器和数据集
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn= RGBTDataloader.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator,
                  drop_last=True), dataset # BatchNorm requires that the number of batches is greater than 1, and drop_last enabled prevents the batch from being 1
