import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,7"
import torchvision
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from utils import *
from tjumnet import Net


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='savedir/epoch_148_psnr33.777_ssim0.970')
parser.add_argument('--savedir', type=str, default='paper/tjumnet')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--dataset', type=str, default='/home/masters/liuyifan2/project/dataset/JUEMR/test/')
args = parser.parse_args()


@torch.no_grad()
def evaluate1(model, loader, save_dir):
    model.eval()
    for image, image_name, _, h, w in tqdm(loader, desc='Inference'):
        if torch.cuda.is_available():
            image = image.cuda()

        image = transforms.Resize((512, 512))(image)
        pred = model(image)
        pred = transforms.Resize((h, w))(pred)
        file_name = os.path.join(save_dir, image_name[0])
        torchvision.utils.save_image(pred.cpu(), file_name)


def main(save_dir):
    dataset = DatasetForInference(args.dataset)
    loader = DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, drop_last=False, shuffle=False,
                        pin_memory=True)

    model = Net()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    if torch.cuda.is_available():
        model = model.cuda()

    evaluate1(model, loader, save_dir)


if __name__ == '__main__':
    main(args.save_dir)
