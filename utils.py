import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import importlib


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


def get_func(path):
    module = path[:path.rfind('.')]
    model_name = path[path.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model_name)
    return net_func