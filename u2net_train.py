import os
import glob
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader


from model import U2NET
from model import U2NETP

from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(reduction='mean')


def multi_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    #     loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item()))

    return loss0, loss


def save_model(model_path, epoch, net, optimizer, loss, tar_loss):
    torch.save({
        'epoch': epoch + 1,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'tar_loss': tar_loss,
    }, model_path)


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="U2NET Segmentation Training", add_help=add_help)

    # File parameters
    parser.add_argument("--data-path", default='/content/data', type=str, help="path to training dataset")
    parser.add_argument("--img-path", default='img', type=str, help="path to training images")
    parser.add_argument("--lbl-path", default='mask', type=str, help="path to training masks")
    parser.add_argument("--save-path", default='/content/drive/MyDrive/Logs/U2NET', type=str, help="path for saving models")

    parser.add_argument("--resume", default='', type=str, help="path to pre-trained models")

    parser.add_argument("--img-ext", default='.*', type=str, help="extension of image files. '.*' if has multiple extensions")
    parser.add_argument("--lbl-ext", default='.jpg', type=str, help="extension of label files")

    # Training hyper-parameters
    parser.add_argument("--epoch", default=1000, type=int, help="number of epochs")
    parser.add_argument("--train-batch", default=12, type=int, help="training batch size")
    parser.add_argument("--val-batch", default=1, type=int, help="validation batch size")
    parser.add_argument("--model", default='u2net', type=str, help="model name: u2net or u2netp")
    parser.add_argument("--worker", default=2, type=int, help="number of workers")

    # Pre-processing parameters
    parser.add_argument("--resize", default=320, type=int, help="rescale size (int or tuple (h, w))")
    parser.add_argument("--crop", default=288, type=int, help="random crop size (int or tuple (h, w))")

    # Optimizer hyper-parameters
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--eps", default=1e-08, type=float, help="epsilon")
    parser.add_argument("--wd", default=0.0, type=float, help="weight decay")

    parser.add_argument("--print-frq", default=50, type=int, help="print log every n iterations")

    return parser


def main():
    # ------- 2. set the directory of training dataset --------
    args = get_args_parser().parse_args()

    model_name = args.model

    model_dir = os.path.join(args.save_path, model_name)
    os.makedirs(model_dir, exist_ok=True)

    epoch_num = args.epoch
    batch_size_train = args.train_batch
    start_epoch = 0

    tra_img_name_list = glob.glob(os.path.join(args.data_path, args.img_path, '*' + args.img_ext))

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]

        bbb = img_name.split(".")[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(os.path.join(args.data_path, args.lbl_path, imidx + args.lbl_ext))

    print("-" * 50)
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("-" * 50)

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(img_name_list=tra_img_name_list,
                                   lbl_name_list=tra_lbl_name_list,
                                   transform=transforms.Compose([
                                       RescaleT(args.resize),
                                       RandomCrop(args.crop),
                                       ToTensorLab(flag=0)]))

    salobj_dataloader = DataLoader(salobj_dataset,
                                   batch_size=batch_size_train,
                                   shuffle=True,
                                   num_workers=args.worker)

    # ------- 3. define model --------
    # define the net
    if model_name == 'u2net':
        net = U2NET(3, 1)
    elif model_name == 'u2netp':
        net = U2NETP(3, 1)
    else:
        net = U2NET(3, 1)

    # ------- 4. define optimizer --------
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=args.eps, weight_decay=args.wd)
    old_loss = 1.0

    if args.resume != '':
        checkpoint = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(checkpoint["model"])

        if torch.cuda.is_available():
            net.cuda()

        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        old_loss = checkpoint["loss"]

    if torch.cuda.is_available():
        net.cuda()

    # ------- 5. training process --------
    print("---Start training---\n")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0

    print_frq = args.print_frq

    for epoch in range(start_epoch, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = multi_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            if ite_num % print_frq == 0:
                print("[INFO] Epoch: %3d/%3d, batch: %5d/%5d, ite: %d, train loss: %3f, tar: %3f " % (
                    epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num,
                    running_loss / ite_num4val, running_tar_loss / ite_num4val))

        print('[INFO] Saving latest model ...')
        save_model(os.path.join(model_dir, model_name + "_latest.pth"),
                   epoch, net, optimizer,
                   running_loss / ite_num4val, running_tar_loss / ite_num4val)
        print('[INFO] Saved latest model')

        if running_loss / ite_num4val < old_loss:
            print('[INFO] Saving best model ...')
            save_model(os.path.join(model_dir, model_name + "_best.pth"),
                       epoch, net, optimizer,
                       running_loss / ite_num4val, running_tar_loss / ite_num4val)
            print('[INFO] Saved best model')

            old_loss = running_loss / ite_num4val
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

        print('[INFO] Epoch: %3d DONE\n' % (epoch + 1))


if __name__ == '__main__':
    main()
