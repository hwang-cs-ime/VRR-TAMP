import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F


def save_checkpoint(state, is_best_miou_sub, is_best_miou_obj, args, filename='default'):
    if filename=='default':
        filename = 'RR-TAmp_%s_batch%d'%(args.dataset,args.batch_size)
    checkpoint_name = './saved_models/%s_checkpoint.pth.tar'%(filename)
    best_name_sub_obj = './saved_models/%s_model_best_sub_obj.pth.tar'%(filename)
    torch.save(state, checkpoint_name)
    if is_best_miou_sub and is_best_miou_obj:
        shutil.copyfile(checkpoint_name, best_name_sub_obj)

def load_pretrain(model, args, logging):
    if os.path.isfile(args.pretrain):
        checkpoint = torch.load(args.pretrain)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert (len([k for k, v in pretrained_dict.items()])!=0)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> loaded pretrain model at {}"
              .format(args.pretrain))
        logging.info("=> loaded pretrain model at {}"
              .format(args.pretrain))
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        print(("=> no pretrained file found at '{}'".format(args.pretrain)))
        logging.info("=> no pretrained file found at '{}'".format(args.pretrain))
    return model

def load_resume(model, args, logging):
    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume)))
        logging.info(("=> no checkpoint found at '{}'".format(args.resume)))
    return model