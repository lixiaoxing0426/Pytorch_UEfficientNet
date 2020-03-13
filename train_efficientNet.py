import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch

sys.path.append('./EfficientNet-PyTorch/')
from efficientnet_pytorch import UEfficientNet

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def train_net(net,
              epochs=50,
              batch_size=1,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              gpu=True,
              img_scale=[513,513]):

    #dir_img = '/home/lixiaoxing/data/DRIVE/train/'
    #dir_mask = '/home/lixiaoxing/data/DRIVE/trainannot/'
    dir_img = '/home/lixiaoxing/github/Pytorch-UNet/data/DRIVE/AV_groundTruth/training/images_jpg/'
    dir_mask = '/home/lixiaoxing/github/Pytorch-UNet/data/DRIVE/AV_groundTruth/training/vessel/'
    dir_checkpoint = 'checkpoints/'
    if os.path.exists(dir_checkpoint) is False:
        os.makedirs(dir_checkpoint)

    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)
        #print(train)

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):

            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            # up features
            #print('**********************up**************************')
            #up_feature = net.extract_features(imgs)
            #print(up_feature.shape)
            #ff = net._blocks[38]._depthwise_conv

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)
            #print(true_masks_flat.shape)
            #print(masks_probs_flat.shape)

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
    parser.add_option('-s', '--scale', dest='scale', type='int',
                      default=[256,256], help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    args.epochs=1000
    args.batchsize=4
    args.gpu="True"
    args.lr=0.0001
    args.scale=[448,448]
    
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
