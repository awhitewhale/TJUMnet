# import package

import time
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "4,7"
import torchvision
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from torchvision import transforms
from utils import *
from tjumnet import Net


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='savedir/latest_model')
parser.add_argument('--dir_path', type=str, default='paper/input')
parser.add_argument('--save_dir', type=str, default='paper/tjumnet')
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()


@torch.no_grad()
def evaluate1(model, loader, dir_path, save_dir):

    start = time.time()
    model.eval()
    for image, image_name, _, h, w in tqdm(loader, desc='Inference'):
        if torch.cuda.is_available():
            image = image.cuda()

        image = transforms.Resize((512, 512))(image)
        pred = model(image)
        pred = transforms.Resize((h, w))(pred)
        file_name = os.path.join(save_dir, image_name[0])
        torchvision.utils.save_image(pred.cpu(), file_name)

@torch.no_grad()
def evaluate(model, loader):
    start = time.time()
    model.eval()
    for image, image_name, _, h, w in tqdm(loader, desc='Inference'):

        if torch.cuda.is_available():
            image = image.cuda()

        pred = model(image)

        file_name = os.path.join(args.save_dir, image_name[0])
        torchvision.utils.save_image(pred.cpu(), file_name)



def main1(dir_path, save_dir):
    dataset_func = dataset.DatasetForInference(args.dataset)

    dataset = dataset_func(dir_path=dir_path)
    loader = DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, drop_last=False, shuffle=False,
                        pin_memory=True)

    model = Net()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    if torch.cuda.is_available():
        model = model.cuda()

    evaluate1(model, loader, dir_path, save_dir)


def main():
    net_func = get_func(args.model)
    dataset_func = get_func(args.dataset)

    dataset = dataset_func(dir_path=args.dir_path)
    loader = DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, drop_last=False, shuffle=False,
                        pin_memory=True)

    model = net_func()


    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    if torch.cuda.is_available():
        model = model.cuda()

    evaluate(model, loader)


if __name__ == '__main__':
    main()
