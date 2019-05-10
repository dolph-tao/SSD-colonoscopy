from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from onestage_detecttracker import build_ssd, ONESTAGE_trackerNet
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from ssd import build_ssd as bbbuild
from tensorboardX import SummaryWriter




def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

writer = SummaryWriter()
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='polypDetection',
                    type=str, help='AnatomyDetection')
parser.add_argument('--dataset_root', default='./dataset/polypDetection/',
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='AnatomyDetection/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)




def train():
    print("start...")
    dataset = DetectionDataset(root=args.dataset_root,
                               transform=SSDAugmentation(300,
                                                         cfg_polyp['mean']), phase='train')
    print(len(dataset))

    ssd_net = build_ssd('train', 300, cfg_polyp['num_classes'])
    net = ssd_net

    if args.cuda:
        torch.cuda.set_device(cfg_polyp['cuda_device'])
        # net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.save_folder+args.resume)
    else:
        # vgg_weights = torch.load(args.save_folder + args.basenet)
        # model_dict = ssd_net.vgg.state_dict()
        # model_dict.pop('0.bias')
        # model_dict.pop('0.weight')

        # vgg_weights = bbbuild('train', 300, cfg['num_classes'])
        vgg_weights = torch.load(args.save_folder +args.basenet)
        model_dict = net.state_dict()

        vgg_weights = {k: v for k, v in vgg_weights.items() if k in model_dict}
        # model_dict = ssd_net.vgg.state_dict()
        model_dict.update(vgg_weights)
        print('Loading base network...')
        net.load_state_dict(model_dict)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        # ssd_net.vgg.apply(weights_init)
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        ssd_net.simi.apply(weights_init)
        # ssd_net.simiSE.apply(weights_init)

    # 只训练simi
    # for param in net.parameters():
    #     param.requires_grad = False
    #
    for param in net.simi.parameters():
        param.requires_grad = False
    # for param in net.simiSE.parameters():
    #     param.requires_grad = False



    # optimizer = optim.SGD(net.simi.parameters(), lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)

    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)

    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    criterion = MultiBoxLoss(num_classes=cfg_polyp['num_classes'],
                             overlap_thresh=0.5,
                             prior_for_matching=True,
                             bkg_label=0,
                             neg_mining=True,
                             neg_pos=3,
                             neg_overlap=0.5,
                             encode_target=False,
                             use_gpu=args.cuda)


    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    simi_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:   polyp Images')
    print('Using the specified args:')
    print(args)

    step_index = 0


    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    # create batch iterator
    batch_iterator = iter(cycle(data_loader))
    for iteration in range(args.start_iter, cfg_polyp['max_iter']):
        # with torch.cuda.device(1):


        if iteration in cfg_polyp['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

            # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            # targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            with torch.no_grad():
                targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c,loss_s = criterion(out, targets)
        loss = loss_l+loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        simi_loss +=loss_s.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f || Loss_c: %.4f || Loss_l: %.4f || Loss_s: %.4f' %
                  (loss.item(), loss_c.item(), loss_l.item(),loss_s.item()))
            writer.add_scalars('polyp_origins/train_from_scratch_ada2',{'loss_c': float(loss_c.item()),
                                              'loss_l': float(loss_l.item()),
                                              'loss_s': float(loss_s.item())}, iteration)



        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'AnatomyDetection/polyp_ssd300adam2_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), 'AnatomyDetection/polyp_ssd300adam2_' +
               repr(iteration) + '.pth')
    writer.close()


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    # init.xavier_uniform(param)
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()



def output():
    print("start...")
    dataset = DetectionDataset(root=args.dataset_root,
                               transform=SSDAugmentation(300,
                                                         cfg_polyp['mean']), phase='test')
    print(len(dataset))

    ssd_net = build_ssd('test', 300, cfg_polyp['num_classes'])
    net = ssd_net

    if args.cuda:
        torch.cuda.set_device(cfg_polyp['cuda_device'])
        # net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    net.load_state_dict(torch.load('AnatomyDetection/Onlysimiloss_ssd300_disease_119999.pth'))
    net.eval()

    img1 = cv2.imread('D:/QTpy_Project/401.jpg')
    transform = BaseTransform(300, (82, 102, 176))
    img = torch.from_numpy(transform(img1)[0]).permute(2, 0, 1)
    img =  img.unsqueeze(0)  # 255也可以改为256
    img = img.cuda()
    outcome = net(img).data
    scale = torch.Tensor([img1.shape[1], img1.shape[0],
                          img1.shape[1], img1.shape[0],1,1,1,1])

    detectionResults = []
    # i是疾病种类，j是框的个数
    for i in range(outcome.size(1)):
        j = 0
        while outcome[0, i, j, 0] >= 0.63:
            score = outcome[0, i, j, 0]
            score = float('%.3f' % score)
            pt = (outcome[0, i, j, 1:] * scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3])
            featureID = (pt[4], pt[5], pt[6], pt[7])
            # print((score, pt))
            newDetection = {'flag': i,
                            'conf': score,
                            'coords': coords,
                            'featureID':featureID
                            }
            detectionResults.append(newDetection)
            j += 1
    pass
if __name__ == '__main__':
    train()
    # output()
