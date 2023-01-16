import os
import sys
import argparse
import time
import logging
import datetime
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data.distributed

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from dataset.data_loader import VGDataset
from models.model import VRR_TAMP
from models.loss import Reg_Loss, GIoU_Loss
from utils.utils import *
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume


def main():
    parser = argparse.ArgumentParser(description='Dataloader test')

    parser.add_argument('--gpu', default='0, 1, 2', help='gpu id')

    parser.add_argument('--workers', default=32, type=int, help='num workers for data loading')

    parser.add_argument('--nb_epoch', default=500, type=int, help='training epoch')

    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

    parser.add_argument('--lr_dec', default=0.1, type=float, help='decline of learning rate')

    parser.add_argument('--batch_size', default=24, type=int, help='batch size')

    parser.add_argument('--size', default=640, type=int, help='image size')

    parser.add_argument('--data_root', type=str, default=' ', help='The root dataset directory')

    parser.add_argument('--dataset', default='vg', type=str, help='vg')

    parser.add_argument('--time', default=10, type=int, help='maximum time steps (lang length) per batch')

    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    parser.add_argument('--pretrain', default='', type=str, metavar='PATH', help='pretrain support load state_dict that are not identical, while have no loss saved as resume')

    parser.add_argument('--print_freq', '-p', default=20, type=int, metavar='N', help='print frequency (default: 1e3)')

    parser.add_argument('--savename', default='VRR-TAMP_1.0', type=str, help='Name head for saved model')

    parser.add_argument('--seed', default=1, type=int, help='random seed')

    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')

    parser.add_argument('--train', dest='train', default=False, action='store_true', help='train')

    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')

    parser.add_argument('--w_div', default=0.125, type=float, help='weight of the diverge loss')

    parser.add_argument('--tunebert', dest='tunebert', default=True, action='store_true', help='if tunebert')

    # DETR
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")

    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help="Disables auxiliary decoding losses (loss at each layer)")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")

    parser.add_argument('--dilation', action='store_true', help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")

    parser.add_argument('--dec_layers', default=0, type=int, help="Number of decoding layers in the transformer")

    parser.add_argument('--dim_feedforward', default=2048, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")

    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")

    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")

    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")

    parser.add_argument('--num_queries', default=400 + 10 + 1, type=int, help="Number of query slots in VLFusion")

    parser.add_argument('--pre_norm', action='store_true')

    global args, anchors_full
    args = parser.parse_args()
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    torch.cuda.manual_seed_all(args.seed + 3)

    # save logs
    if args.savename == 'default':
        args.savename = 'VRR-TAMP_%s_batch%d' % (args.dataset, args.batch_size)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename="./logs/%s" % args.savename, filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv))
    logging.info(str(args))

    input_transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = VGDataset(data_root=args.data_root, dataset=args.dataset, split='train', imsize=args.size, transform=input_transform, max_query_len=args.time)
    val_dataset = VGDataset(data_root=args.data_root, dataset=args.dataset, split='val', imsize=args.size, max_query_len=args.time, transform=input_transform)
    test_dataset = VGDataset(data_root=args.data_root, dataset=args.dataset, testmode=True, split='val', imsize=args.size, max_query_len=args.time, transform=input_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True, num_workers=0)

    # Model
    model = VRR_TAMP(bert_model=args.bert_model, tunebert=args.tunebert, args=args)
    model = torch.nn.DataParallel(model).cuda()

    if args.pretrain:
        model = load_pretrain(model, args, logging)
    if args.resume:
        model = load_resume(model, args, logging)

    print('The total num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('The total num of parameters:%d' % int(sum([param.nelement() for param in model.parameters()])))

    if args.tunebert:
        visu_param = model.module.visumodel.parameters()
        text_param = model.module.textmodel.parameters()
        rest_param = [param for param in model.parameters() if
                      ((param not in visu_param) and (param not in text_param))]
        visu_param = list(model.module.visumodel.parameters())
        text_param = list(model.module.textmodel.parameters())
        sum_visu = sum([param.nelement() for param in visu_param])
        sum_text = sum([param.nelement() for param in text_param])
        sum_fusion = sum([param.nelement() for param in rest_param])
        print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)
    else:
        visu_param = model.module.visumodel.parameters()
        rest_param = [param for param in model.parameters() if param not in visu_param]
        visu_param = list(model.module.visumodel.parameters())
        sum_visu = sum([param.nelement() for param in visu_param])
        sum_text = sum([param.nelement() for param in model.module.textmodel.parameters()])
        sum_fusion = sum([param.nelement() for param in rest_param]) - sum_text
        print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)

    # optimizer
    if args.tunebert:
        optimizer = torch.optim.AdamW([{'params': rest_param}, {'params': visu_param, 'lr': args.lr / 10.}, {'params': text_param, 'lr': args.lr / 10.}], lr=args.lr, weight_decay=0.0001)
    else:
        optimizer = torch.optim.AdamW([{'params': rest_param}, {'params': visu_param}], lr=args.lr, weight_decay=0.0001)

    # training and testing
    best_accu_sub = -float('Inf')
    best_accu_obj = -float('Inf')
    best_miou_sub = -float('Inf')
    best_miou_obj = -float('Inf')
    if args.train:
        for epoch in range(0, args.nb_epoch):
            adjust_learning_rate(args, optimizer, epoch)

            print('visual_lr {visual_lr:.8f}\t   language_lr {language_lr:.8f}\t' \
                .format(visual_lr=optimizer.param_groups[1]['lr'], language_lr=optimizer.param_groups[2]['lr']))

            logging.info('visual_lr {visual_lr:.8f}\t    language_lr {language_lr:.8f}\t' \
                .format(visual_lr=optimizer.param_groups[1]['lr'], language_lr=optimizer.param_groups[2]['lr']))

            train_epoch(train_loader, model, optimizer, epoch)
            accu_new_sub, accu_new_obj, miou_new_sub, miou_new_obj = validate_epoch(val_loader, model)

            # remember best accu and save checkpoint
            is_best_accu_sub = accu_new_sub >= best_accu_sub
            best_accu_sub = max(accu_new_sub, best_accu_sub)

            is_best_accu_obj = accu_new_obj >= best_accu_obj
            best_accu_obj = max(accu_new_obj, best_accu_obj)

            is_best_miou_sub = miou_new_sub >= best_miou_sub
            best_miou_sub = max(miou_new_sub, best_miou_sub)

            is_best_miou_obj = miou_new_obj >= best_miou_obj
            best_miou_obj = max(miou_new_obj, best_miou_obj)

            save_checkpoint({'epoch': epoch + 1, 'lr': args.lr, 'state_dict': model.state_dict(), 'accu_new_sub': accu_new_sub,
                 'accu_new_obj': accu_new_obj, 'miou_new_sub': miou_new_sub, 'miou_new_obj': miou_new_obj,
                 'optimizer': optimizer.state_dict(), }, is_best_miou_sub, is_best_miou_obj, args, filename=args.savename)

        print('\nBest Accu_sub: %f, Best Accu_obj: %f, Best miou_sub: %f, Best miou_obj: %f\n' % (best_accu_sub, best_accu_obj, is_best_miou_sub, is_best_miou_obj))
        logging.info('\nBest Accu_sub: %f, Best Accu_obj: %f, Best miou_sub: %f, Best miou_obj: %f\n' % (best_accu_sub, best_accu_obj, is_best_miou_sub, is_best_miou_obj))
        logging.info(
            "\n optimizer.param_groups[0]['lr']={}, optimizer.param_groups[1]['lr']={}, optimizer.param_groups[2]['lr']={}\n".format(
                optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[2]['lr']))


def train_epoch(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l1_losses_sub = AverageMeter()
    l1_losses_obj = AverageMeter()
    GIoU_losses_sub = AverageMeter()
    GIoU_losses_obj = AverageMeter()
    acc_sub = AverageMeter()
    acc_obj = AverageMeter()
    miou_sub = AverageMeter()
    miou_obj = AverageMeter()

    # Sets the module in training mode
    model.train()
    end = time.time()

    for batch_idx, (imgs, masks, word_id, word_mask, gt_bbox_sub, gt_bbox_obj) in enumerate(train_loader):
        # 12 x 3 x 640 x 640
        imgs = imgs.cuda()
        # 12 x 640 x 640 x 3
        masks = masks.cuda()
        masks = masks[:, :, :, 0] == 255
        # 12 x 40
        word_id = word_id.cuda()
        # 12 x 40
        word_mask = word_mask.cuda()
        # 12 x 4
        gt_bbox_sub = gt_bbox_sub.cuda()
        gt_bbox_obj = gt_bbox_obj.cuda()
        image = Variable(imgs)
        masks = Variable(masks)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        gt_bbox_sub = Variable(gt_bbox_sub)
        gt_bbox_sub = torch.clamp(gt_bbox_sub, min=0, max=args.size - 1)
        gt_bbox_obj = Variable(gt_bbox_obj)
        gt_bbox_obj = torch.clamp(gt_bbox_obj, min=0, max=args.size - 1)

        pred_bbox_sub, pred_bbox_obj = model(image, masks, word_id, word_mask)

        # compute loss
        loss = 0.
        GIoU_loss_sub = GIoU_Loss(pred_bbox_sub * (args.size - 1), gt_bbox_sub, args.size - 1)
        GIoU_loss_obj = GIoU_Loss(pred_bbox_obj * (args.size - 1), gt_bbox_obj, args.size - 1)
        loss = loss + GIoU_loss_sub + GIoU_loss_obj

        gt_bbox_sub_ = xyxy2xywh(gt_bbox_sub)
        gt_bbox_obj_ = xyxy2xywh(gt_bbox_obj)
        l1_loss_sub = Reg_Loss(pred_bbox_sub, gt_bbox_sub_ / (args.size - 1))
        l1_loss_obj = Reg_Loss(pred_bbox_obj, gt_bbox_obj_ / (args.size - 1))
        loss = loss + l1_loss_sub + l1_loss_obj

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), imgs.size(0))
        l1_losses_sub.update(l1_loss_sub.item(), imgs.size(0))
        l1_losses_obj.update(l1_loss_obj.item(), imgs.size(0))

        GIoU_losses_sub.update(GIoU_loss_sub.item(), imgs.size(0))
        GIoU_losses_obj.update(GIoU_loss_obj.item(), imgs.size(0))

        # subject box iou
        pred_bbox_sub = torch.cat([pred_bbox_sub[:, :2] - (pred_bbox_sub[:, 2:] / 2), pred_bbox_sub[:, :2] + (pred_bbox_sub[:, 2:] / 2)], dim=1)
        pred_bbox_sub = pred_bbox_sub * (args.size - 1)
        iou_sub = bbox_iou(pred_bbox_sub.data.cpu(), gt_bbox_sub.data.cpu(), x1y1x2y2=True)
        accu_sub = np.sum(np.array((iou_sub.data.cpu().numpy() > 0.5), dtype=float)) / args.batch_size

        # object box iou
        pred_bbox_obj = torch.cat([pred_bbox_obj[:, :2] - (pred_bbox_obj[:, 2:] / 2), pred_bbox_obj[:, :2] + (pred_bbox_obj[:, 2:] / 2)], dim=1)
        pred_bbox_obj = pred_bbox_obj * (args.size - 1)
        iou_obj = bbox_iou(pred_bbox_obj.data.cpu(), gt_bbox_obj.data.cpu(), x1y1x2y2=True)
        accu_obj = np.sum(np.array((iou_obj.data.cpu().numpy() > 0.5), dtype=float)) / args.batch_size

        # metrics
        miou_sub.update(torch.mean(iou_sub).item(), imgs.size(0))
        acc_sub.update(accu_sub, imgs.size(0))

        # metrics
        miou_obj.update(torch.mean(iou_obj).item(), imgs.size(0))
        acc_obj.update(accu_obj, imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 10000 == 0:
            print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'L1_Loss_sub {l1_loss_sub.val:.4f} ({l1_loss_sub.avg:.4f})\t' \
                        'L1_Loss_obj {l1_loss_obj.val:.4f} ({l1_loss_obj.avg:.4f})\t' \
                        'GIoU_Loss_sub {GIoU_loss_sub.val:.4f} ({GIoU_loss_sub.avg:.4f})\t' \
                        'GIoU_Loss_obj {GIoU_loss_obj.val:.4f} ({GIoU_loss_obj.avg:.4f})\t' \
                        'Accu_sub {acc_sub.val:.4f} ({acc_sub.avg:.4f})\t' \
                        'Accu_obj {acc_obj.val:.4f} ({acc_obj.avg:.4f})\t' \
                        'Mean_iou_sub {miou_sub.val:.4f} ({miou_sub.avg:.4f})\t' \
                        'Mean_iou_obj {miou_obj.val:.4f} ({miou_obj.avg:.4f})\t' \
                .format(epoch, batch_idx, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, l1_loss_sub=l1_losses_sub,
                        l1_loss_obj=l1_losses_obj, GIoU_loss_sub=GIoU_losses_sub,
                        GIoU_loss_obj=GIoU_losses_obj, miou_sub=miou_sub,
                        miou_obj=miou_obj, acc_sub=acc_sub, acc_obj=acc_obj)
            print(print_str)
            logging.info(print_str)


def validate_epoch(val_loader, model, mode='val'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_sub = AverageMeter()
    acc_obj = AverageMeter()
    miou_sub = AverageMeter()
    miou_obj = AverageMeter()

    model.eval()
    end = time.time()
    print(datetime.datetime.now())

    for batch_idx, (imgs, masks, word_id, word_mask, gt_bbox_sub, gt_bbox_obj) in enumerate(val_loader):
        imgs = imgs.cuda()
        masks = masks.cuda()
        masks = masks[:, :, :, 0] == 255
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()

        gt_bbox_sub = gt_bbox_sub.cuda()
        gt_bbox_obj = gt_bbox_obj.cuda()

        image = Variable(imgs)
        masks = Variable(masks)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)

        gt_bbox_sub = Variable(gt_bbox_sub)
        gt_bbox_sub = torch.clamp(gt_bbox_sub, min=0, max=args.size - 1)

        gt_bbox_obj = Variable(gt_bbox_obj)
        gt_bbox_obj = torch.clamp(gt_bbox_obj, min=0, max=args.size - 1)

        with torch.no_grad():
            pred_bbox_sub, pred_bbox_obj = model(image, masks, word_id, word_mask)

        pred_bbox_sub = torch.cat([pred_bbox_sub[:, :2] - (pred_bbox_sub[:, 2:] / 2), pred_bbox_sub[:, :2] + (pred_bbox_sub[:, 2:] / 2)], dim=1)
        pred_bbox_sub = pred_bbox_sub * (args.size - 1)

        pred_bbox_obj = torch.cat([pred_bbox_obj[:, :2] - (pred_bbox_obj[:, 2:] / 2), pred_bbox_obj[:, :2] + (pred_bbox_obj[:, 2:] / 2)], dim=1)
        pred_bbox_obj = pred_bbox_obj * (args.size - 1)

        # metrics
        iou_sub = bbox_iou(pred_bbox_sub.data.cpu(), gt_bbox_sub.data.cpu(), x1y1x2y2=True)
        accu_sub = np.sum(np.array((iou_sub.data.cpu().numpy() > 0.5), dtype=float)) / args.batch_size

        iou_obj = bbox_iou(pred_bbox_obj.data.cpu(), gt_bbox_obj.data.cpu(), x1y1x2y2=True)
        accu_obj = np.sum(np.array((iou_obj.data.cpu().numpy() > 0.5), dtype=float)) / args.batch_size

        acc_sub.update(accu_sub, imgs.size(0))
        acc_obj.update(accu_obj, imgs.size(0))
        miou_sub.update(torch.mean(iou_sub).item(), imgs.size(0))
        miou_obj.update(torch.mean(iou_obj).item(), imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if batch_idx % args.print_freq == 0:
        if batch_idx % 5000 == 0:
            print_str = '[{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Accu_sub {acc_sub.val:.4f} ({acc_sub.avg:.4f})\t' \
                        'Accu_obj {acc_obj.val:.4f} ({acc_obj.avg:.4f})\t' \
                        'Mean_iou_sub {miou_sub.val:.4f} ({miou_sub.avg:.4f})\t' \
                        'Mean_iou_obj {miou_obj.val:.4f} ({miou_obj.avg:.4f})\t' \
                .format( \
                batch_idx, len(val_loader), batch_time=batch_time, \
                data_time=data_time, \
                acc_sub=acc_sub, acc_obj=acc_obj, miou_sub=miou_sub, miou_obj=miou_obj)
            print(print_str)
            logging.info(print_str)
    print("acc_sub.avg={}, acc_obj.avg={}, miou_sub.avg={}, miou_obj.avg={}".format(acc_sub.avg, acc_obj.avg, miou_sub.avg, miou_obj.avg))
    logging.info("acc_sub.avg=%f, acc_obj.avg=%f, miou_sub.avg=%f, miou_obj.avg=%f" % (acc_sub.avg, acc_obj.avg, float(miou_sub.avg), float(miou_obj.avg)))
    return acc_sub.avg, acc_obj.avg, miou_sub.avg, miou_obj.avg


if __name__ == "__main__":
    main()
