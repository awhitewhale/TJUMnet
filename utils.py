import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from torch.utils.data import Dataset
import json
from torchvision import transforms
import os, random
from glob import glob
from PIL import Image


class Collate():
    def __init__(self, n_degrades) -> None:
        self.n_degrades = n_degrades

    def __call__(self, batch):

        target_images = [[] for _ in range(self.n_degrades)]
        input_images = [[] for _ in range(self.n_degrades)]

        for i in range(len(batch)):
            target_image, input_image, dataset_label = batch[i]
            target_images[dataset_label].append(target_image.unsqueeze(0))
            input_images[dataset_label].append(input_image.unsqueeze(0))

        for i in range(len(target_images)):
            if target_images[i] == []:
                return None, None
            target_images[i] = torch.cat(target_images[i])
            input_images[i] = torch.cat(input_images[i])
        target_images = torch.cat(target_images)

        return target_images, input_images

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_meter(num_meters):
    return [AverageMeter() for _ in range(num_meters)]


def update_meter(meters, values, n=1):
    for meter, value in zip(meters, values):
        meter.update(value, n=n)

    return meters

@torch.no_grad()
def torchPSNR(prd_img, tar_img):
    if not isinstance(prd_img, torch.Tensor):
        prd_img = torch.from_numpy(prd_img)
        tar_img = torch.from_numpy(tar_img)

    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20 * torch.log10(1/rmse)
    return ps

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class Collate():
    def __init__(self, n_degrades) -> None:
        self.n_degrades = n_degrades

    def __call__(self, batch):

        target_images = [[] for _ in range(self.n_degrades)]
        input_images = [[] for _ in range(self.n_degrades)]

        for i in range(len(batch)):
            target_image, input_image, dataset_label = batch[i]
            target_images[dataset_label].append(target_image.unsqueeze(0))
            input_images[dataset_label].append(input_image.unsqueeze(0))

        for i in range(len(target_images)):
            if target_images[i] == []:
                return None, None
            target_images[i] = torch.cat(target_images[i])
            input_images[i] = torch.cat(input_images[i])
        target_images = torch.cat(target_images)

        return target_images, input_images


class DatasetForTrain(Dataset):
    def __init__(self, meta_path):
        self.datasets = []
        self.file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(meta_path + 'GT/') for file in files]
        for file_path in self.file_paths:
            with open(file_path, "r") as f:
                self.datasets.append(file_path)
                self.image_size = 224
                self.transform = transforms.ToTensor()
                self.resize = transforms.Resize((self.image_size, self.image_size))
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        dataset_label = 0
        path = self.datasets
        target_path, input_path = path[index], path[index].replace('GT', 'heavy_snow')
        target_image = Image.open(target_path).convert('RGB')
        input_image = Image.open(input_path).convert('RGB')

        target_image = self.transform(target_image)
        input_image = self.transform(input_image)

        target_image, input_image = self.rand_crop(target_image, input_image)
        target_image, input_image = self.rand_flip(target_image, input_image)

        return target_image, input_image, dataset_label

    def rand_flip(self, target_image, input_image):
        if random.random() > 0.5:
            target_image = target_image.flip(2)
            input_image = input_image.flip(2)

        return target_image, input_image

    def rand_crop(self, target_image, input_image):
        h, w = target_image.shape[1], target_image.shape[2]
        if h < self.image_size or w < self.image_size:
            return self.resize(input_image), self.resize(target_image)

        rr = random.randint(0, h - self.image_size)
        cc = random.randint(0, w - self.image_size)

        target_image = target_image[:, rr: rr + self.image_size, cc: cc + self.image_size]
        input_image = input_image[:, rr: rr + self.image_size, cc: cc + self.image_size]

        return target_image, input_image

class DatasetForValid(Dataset):
    def __init__(self, meta_path):
        self.dataset = []
        self.file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(meta_path + 'GT/') for file in files]
        for file_path in self.file_paths:
            self.dataset.append(file_path)
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.dataset
        target_path, input_path = path[index], path[index].replace('GT', 'heavy_snow')

        target_image = Image.open(target_path).convert('RGB')
        input_image = Image.open(input_path).convert('RGB')

        target_image = self.transform(target_image)
        input_image = self.transform(input_image)

        _, h, w = target_image.shape
        if (h % 16 != 0) or (w % 16 != 0):
            target_image = transforms.Resize(((h // 16) * 16, (w // 16) * 16))(target_image)
            input_image = transforms.Resize(((h // 16) * 16, (w // 16) * 16))(input_image)

        return target_image, input_image


class DatasetForInference(Dataset):
    def __init__(self, dir_path):
        self.image_paths = glob(os.path.join(dir_path + 'heavy_snow/', '*'))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        input_path = self.image_paths[index]
        input_image = Image.open(input_path).convert('RGB')
        input_image = self.transform(input_image)

        _, h, w = input_image.shape
        return input_image, os.path.basename(input_path), _, h, w