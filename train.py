import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse, os, random, sys
from tqdm import tqdm
from utils import Collate
from tjumnet import TFO
from tjumnet import DFCCLoss, MDCCLoss
from tjumnet import Net
from utils import get_meter, update_meter, torchPSNR, ssim, DatasetForTrain, DatasetForValid
import pandas as pd
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
sys.path.append('../../')


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='/home/masters/liuyifan2/project/dataset/JUEMR/train/')
    parser.add_argument('--test', type=str, default='/home/masters/liuyifan2/project/dataset/JUEMR/test/')
    parser.add_argument('--savedir', type=str, default='savedir/')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--warmup_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    return parser.parse_args()
args=arg()

@torch.no_grad()
def evaluate(model, val_loader, epoch):
    psnr_list, ssim_list = [], []
    model.eval()
    for target, image in tqdm(val_loader):
        image = image.cuda()
        target = target.cuda()
        pred = model(image)
        # onnx_net_path = '/home/masters/liuyifan2/project/underwater/savedir/lastmodel.onxx'
        # torch.onnx.export(model, image, onnx_net_path, verbose=False)
        psnr_list.append(torchPSNR(pred, target).item())
        ssim_list.append(ssim(pred, target).item())
    combined_data = list(zip([epoch] * len(psnr_list), psnr_list, ssim_list))
    df = pd.DataFrame(combined_data, columns=['Epoch', 'PSNR', 'SSIM'])
    df.to_csv('log/{}'.format(model.__name__), index=False)
    return np.mean(psnr_list), np.mean(ssim_list)


def train_tfo(model, teacher_networks, TFO, train_loader, optimizer, criterions):
    criterion_l1, criterion_scr, _ = criterions
    model.train()
    TFO.train()
    for teacher_network in teacher_networks:
        teacher_network.eval()

    for target_images, input_images in tqdm(train_loader):
        target_images = target_images.cuda()
        input_images = [images.cuda() for images in input_images]
        preds_from_teachers = []
        features_from_each_teachers = []
        with torch.no_grad():
            for i in range(len(teacher_networks)):
                preds, features = teacher_networks[i](input_images[i], return_feat=True)
                preds_from_teachers.append(preds)
                features_from_each_teachers.append(features)
        preds_from_teachers = torch.cat(preds_from_teachers)
        features_from_teachers = []
        for layer in range(len(features_from_each_teachers[0])):
            features_from_teachers.append([features_from_each_teachers[i][layer] for i in range(len(teacher_networks))])
        preds_from_student, features_from_student = model(torch.cat(input_images), return_feat=True)
        PFE_loss, PFV_loss = 0., 0.
        for i, (s_features, t_features) in enumerate(zip(features_from_student, features_from_teachers)):
            t_proj_features, t_recons_features, s_proj_features = TFO[i](t_features, s_features)
            PFE_loss += criterion_l1(s_proj_features, torch.cat(t_proj_features))
            PFV_loss += 0.05 * criterion_l1(torch.cat(t_recons_features), torch.cat(t_features))
        T_loss = criterion_l1(preds_from_student, preds_from_teachers)
        SCR_loss = 0.1 * criterion_scr(preds_from_student, target_images, torch.cat(input_images))
        total_loss = T_loss + PFE_loss + PFV_loss + SCR_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


def train_hr(model, train_loader, optimizer, criterions):
    criterion_l1, _, criterion_hcr = criterions
    model.train()
    for target_images, input_images in tqdm(train_loader):
        target_images = target_images.cuda()
        input_images = torch.cat(input_images).cuda()
        preds = model(input_images, return_feat=False)
        G_loss = criterion_l1(preds, target_images)
        HCR_loss = 0.2 * criterion_hcr(preds, target_images, input_images)
        total_loss = G_loss + HCR_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        meters = update_meter(meters, [total_loss.item(), G_loss.item(), HCR_loss.item()])




def main():
    random_seed = 19961126
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    teacher_networks = []

    checkpoint = torch.load('teacher/JUEMR_T.pth')
    teacher = Net().cuda()
    teacher.load_state_dict(checkpoint['state_dict'], strict=True)
    teacher_networks.append(teacher)

    train_dataset = DatasetForTrain(args.train)
    val_dataset = DatasetForValid(args.test)
    train_loader = DataLoader(dataset=train_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                              drop_last=True, shuffle=True, collate_fn=Collate(n_degrades=len(teacher_networks)))
    val_loader = DataLoader(dataset=val_dataset, num_workers=args.num_workers, batch_size=1, drop_last=False,
                            shuffle=False)

    tfo_model = nn.ModuleList([])
    for c in [64, 128, 256, 256]:
        tfo_model.append(TFO(channel_t=c, channel_s=c, channel_h=c // 2, n_teachers=len(teacher_networks)))
    tfo_model = tfo_model.cuda()

    criterions = nn.ModuleList([nn.L1Loss(), DFCCLoss(), MDCCLoss()]).cuda()

    model = Net().cuda()

    linear_scaled_lr = args.lr * args.batch_size / 16
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': tfo_model.parameters()}],
                                 lr=linear_scaled_lr, betas=(0.9, 0.999), eps=1e-8)


    for epoch in tqdm(range(args.epoch)):
        if epoch <= 150:
            train_tfo(model, teacher_networks, tfo_model, train_loader, optimizer, criterions)
        else:
            train_hr(model, train_loader, optimizer, criterions)

        if epoch % 10 == 0:
            psnr, ssim = evaluate(model, val_loader, epoch)
            print(psnr, ssim)

        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'ckt_module': tfo_model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(args.save_dir, 'latest_model'))


if __name__ == '__main__':
    main()
