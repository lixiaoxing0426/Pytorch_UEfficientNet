import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from dice_loss import DiceCoeff

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, get_imgs_and_masks_y, batch

sys.path.append('./EfficientNet-PyTorch/')
from efficientnet_pytorch import UEfficientNet

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              #val_percent=0.1,
              save_cp=True,
              gpu=False,
              img_scale=0.5):

    img_train = '/home/lixiaoxing/github/Pytorch-UNet/data/DRIVE/AV_groundTruth/training/images_jpg/'
    mask_train = '/home/lixiaoxing/github/Pytorch-UNet/data/DRIVE/AV_groundTruth/training/vessel/'
    img_val = '/home/lixiaoxing/github/Pytorch-UNet/data/DRIVE/AV_groundTruth/training/val_jpg/'
    mask_val = '/home/lixiaoxing/github/Pytorch-UNet/data/DRIVE/AV_groundTruth/training/val_vessel/'
    dir_checkpoint = 'checkpoints_drive3_adam/'
    if os.path.exists(dir_checkpoint) is False:
        os.makedirs(dir_checkpoint)

    ids_train = get_ids(img_train)
    data_train = split_ids(ids_train)
    data_train = list(data_train)
    ids_val = get_ids(img_val)
    data_val = split_ids(ids_val)
    data_val = list(data_val)

    #iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(data_train),
               len(data_val), str(save_cp), str(gpu)))

    N_train = len(data_train)

    #optimizer = optim.SGD(net.parameters(),
    #                      lr=lr,
    #                      momentum=0.9,
    #                      weight_decay=0.0005)

    optimizer = optim.Adam(net.parameters(),
                           lr=lr,
                           weight_decay=1e-5)
         
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,mode='min',patience=3,verbose=True)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    criterion = nn.BCELoss()
    #criterion = DiceCoeff()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        train = get_imgs_and_masks_y(data_train, img_train, mask_train, img_scale)
        val = get_imgs_and_masks_y(data_val, img_val, mask_val, img_scale)

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            #print(masks_pred.shape, true_masks.shape)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            #print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))

        scheduler.step(val_dice)


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    args.epochs=200
    args.batchsize=3
    args.gpu="True"
    args.lr=0.0001
    args.scale=0.95    #drive
    #args.scale=0.5   # size 900*900

    net = UEfficientNet.from_name('efficientnet-b5')

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
