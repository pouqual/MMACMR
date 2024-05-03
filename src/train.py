# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Training script
"""
import os
import random
from collections import defaultdict
import numpy as np
import pickle
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.cuda.amp import GradScaler, autocast
import gc
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
from utils.loss import TripletLoss
from dataset import get_loader
from config import get_args
from modelsb import get_model
from eval import computeAverageMetrics
import datetime
from knn_features import knn, angular_loss

#torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAP_LOC = None if torch.cuda.is_available() else 'cpu'
print("?????GPU????", torch.cuda.current_device())

def trainIter(args, split, loader, model, optimizer,
              loss_function, scaler, metrics_epoch):
    with autocast():
        #img, title, ingrs, instrs, _ = loader.next()
        img, title, ingrs, instrs, _ = next(loader)

        img = img.to(device) if img is not None else None
        title = title.to(device)
        ingrs = ingrs.to(device)
        instrs = instrs.to(device)


        if split == 'val':
            with torch.no_grad():
                img_feat, recipe_feat = model(img, title, ingrs, instrs)
        else:
            out = model(img, title, ingrs, instrs,
                        freeze_backbone=args.freeze_backbone)

            img_feat, recipe_feat = out       #recipe_feat(128,1024)

        idxs, dists = knn(recipe_feat)


        loss_recipe, loss_paired, loss_knn = 0, 0, 0


        neigbors = recipe_feat[idxs[0][1]]      #我也忘了为啥这么写，先取第一个，再cat后面的
        neigbors = neigbors.unsqueeze(0)
        negas = recipe_feat[idxs[0][random.randint(12,len(recipe_feat)-1)]]
        negas = negas.unsqueeze(0)
        for i in range(1,len(recipe_feat)):      #该batch内的每个
            neigbors = torch.cat((neigbors,recipe_feat[idxs[i][1]].unsqueeze(0)),dim=0)
            negas = torch.cat((negas,recipe_feat[idxs[i][random.randint(12,len(recipe_feat)-1)]].unsqueeze(0)),dim=0)

        neigbors0 = img_feat[idxs[0][1]]
        neigbors0 = neigbors0.unsqueeze(0)
        negas0 = img_feat[idxs[0][random.randint(12, len(img_feat) - 1)]]
        negas0 = negas0.unsqueeze(0)
        for i in range(1, len(img_feat)):  # 该batch内的每个
            neigbors0 = torch.cat((neigbors0, img_feat[idxs[i][1]].unsqueeze(0)), dim=0)
            negas0 = torch.cat((negas0, img_feat[idxs[i][random.randint(12, len(img_feat) - 1)]].unsqueeze(0)),dim=0)

        '''
            idx = idxs[i][1:11]
            nega = []
            for j in range(len(idx)):
                neigbors.append(ing_feat[j])        #最近的10个的ing_feat
            k = random.randint(12,len(ing_feat)-1)
            nega.append(ing_feat[k])
        '''

        loss_text = angular_loss(anchors=recipe_feat, positives=neigbors, negatives=negas)
        loss_image = angular_loss(anchors=img_feat, positives=neigbors0, negatives=negas0)
        loss_knn = 0.09 * loss_text + 0.1 * loss_image

        '''
        positive_projections = recipe_feat
        neighbor_projections = neigbors
        negative_projections = negas

        positive_projections_np = torch.repeat_interleave(positive_projections, len(negative_projections), dim=0)
        neighbor_projections_np = torch.repeat_interleave(neighbor_projections, len(negative_projections), dim=0)
        negative_projections_np = negative_projections.repeat(len(negative_projections), 1)
        loss_knn = angular_loss(anchors=positive_projections_np, positives=neighbor_projections_np, negatives=negative_projections_np)
        '''


        if img is not None:
            loss_paired = loss_function(img_feat, recipe_feat)
            metrics_epoch['loss_paired'].append(loss_paired.item())


        loss = loss_paired  + 0.01 * loss_knn

        metrics_epoch['loss'].append(loss.item())

    if split == 'train':
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if img_feat is not None:
        img_feat = img_feat.cpu().detach().numpy()
    recipe_feat = recipe_feat.cpu().detach().numpy()

    return img_feat, recipe_feat, metrics_epoch


def train(args):

    checkpoints_dir = os.path.join(args.save_dir, args.model_name)
    make_dir(checkpoints_dir)

    if args.tensorboard:
        logger = SummaryWriter(checkpoints_dir)

    loaders = {}

    if args.resume_from != '':
        print("Resuming from checkpoint: ", args.resume_from)
        # update arguments when loading model
        vars_to_replace = ['batch_size', 'tensorboard',
                           'model_name', 'lr',
                           'scale_lr', 'freeze_backbone',
                           'load_optimizer']
        store_dict = {}
        for var in vars_to_replace:
            store_dict[var] = getattr(args, var)

        resume_path = os.path.join(args.save_dir, args.resume_from)
        args, model_dict, optim_dict = load_checkpoint(resume_path,
                                                       'curr', MAP_LOC,
                                                       store_dict)

        # load current state of training
        curr_epoch = args.curr_epoch
        best_loss = args.best_loss

        for var in vars_to_replace:
            setattr(args, var, store_dict[var])
    else:
        curr_epoch = 0
        best_loss = np.inf
        model_dict, optim_dict = None, None

    for split in ['train', 'val']:
        loader, dataset = get_loader(args.root, args.batch_size, args.resize,
                                     args.imsize,
                                     augment=True,
                                     split=split, mode=split,
                                     text_only_data=False)
        loaders[split] = loader

    # create dataloader for training samples without images
    use_extra_data = True
    if args.recipe_loss_weight > 0 and use_extra_data:
        loader_textonly, _ = get_loader(args.root, args.batch_size*2,
                                        args.resize,
                                        args.imsize,
                                        augment=True,
                                        split='train', mode='train',
                                        text_only_data=True)

    vocab_size = len(dataset.get_vocab())
    model = get_model(args, vocab_size)

    params_backbone = list(model.image_encoder.backbone.parameters())
    params_fc = list(model.image_encoder.fc.parameters()) \
                + list(model.text_encoder.parameters()) \
                + list(model.merger_recipe.parameters())
    '''params_fc = list(model.image_encoder.fc.parameters()) \
                + list(model.text_encoder.parameters()) \
                + list(model.text_encoder1.parameters()) \
                + list(model.merger_recipe.parameters())'''

    print("recipe encoder", count_parameters(model.text_encoder))
    #print("recipe encoder1", count_parameters(model.text_encoder1))
    print("image encoder", count_parameters(model.image_encoder))

    optimizer = get_optimizer(params_fc,
                              params_backbone,
                              args.lr, args.scale_lr, args.wd,
                              freeze_backbone=args.freeze_backbone)

    if model_dict is not None:
        model.load_state_dict(model_dict)
        if args.load_optimizer:
            try:
                optimizer.load_state_dict(optim_dict)
            except:
                print("Could not load optimizer state. Using default initialization...")

    ngpus = 1
    if device != 'cpu' and torch.cuda.device_count() > 1:
        ngpus = torch.cuda.device_count()
        model = nn.DataParallel(model)
        #model = nn.DataParallel(model, device_ids=[1])
        print("?????GPU????", torch.cuda.current_device())

    model = model.to(device)

    if device != 'cpu':
        cudnn.benchmark = True

    # learning rate scheduler
    scheduler = get_scheduler(args, optimizer)

    loss_function = TripletLoss(margin=args.margin)

    # training loop
    wait = 0

    scaler = GradScaler()

    for epoch in range(curr_epoch, args.n_epochs):

        for split in ['train', 'val']:
            if split == 'train':
                model.train()
            else:
                model.eval()

            metrics_epoch = defaultdict(list)

            total_step = len(loaders[split])
            loader = iter(loaders[split])

            if args.recipe_loss_weight > 0 and use_extra_data:
                iterator_textonly = iter(loader_textonly)

            img_feats, recipe_feats = None, None

            emult = 2 if (args.recipe_loss_weight > 0 and use_extra_data and split == 'train') else 1

            for i in range(total_step*emult):

                # sample from paired or text-only data loaders - only do this for training
                if i%2 == 0 and emult == 2:
                    this_iter_loader = iterator_textonly
                else:
                    this_iter_loader = loader

                optimizer.zero_grad()
                model.zero_grad()
                img_feat, recipe_feat, metrics_epoch = trainIter(args, split,
                                                                 this_iter_loader,
                                                                 model, optimizer,
                                                                 loss_function,
                                                                 scaler,
                                                                 metrics_epoch)

                if img_feat is not None:
                    if img_feats is not None:
                        img_feats = np.vstack((img_feats, img_feat))
                        recipe_feats = np.vstack((recipe_feats, recipe_feat))
                    else:
                        img_feats = img_feat
                        recipe_feats = recipe_feat

                if not args.tensorboard and i != 0 and i % args.log_every == 0:
                    # log metrics to stdout every few iterations
                    avg_metrics = {k: np.mean(v) for k, v in metrics_epoch.items() if v}
                    text_ = "split: {:s}, epoch [{:d}/{:d}], step [{:d}/{:d}]"
                    values = [split, epoch, args.n_epochs, i, total_step]
                    for k, v in avg_metrics.items():
                        text_ += ", " + k + ": {:.4f}"
                        values.append(v)
                    str_ = text_.format(*values)
                    print(str_)

            # computes retrieval metrics (average of 10 runs on 1k sized rankings)
            retrieval_metrics = computeAverageMetrics(img_feats, recipe_feats,
                                                      1000, 10, forceorder=True)

            for k, v in retrieval_metrics.items():
                metrics_epoch[k] = v

            avg_metrics = {k: np.mean(v) for k, v in metrics_epoch.items() if v}
            # log to stdout at the end of the epoch (for both train and val splits)
            if not args.tensorboard:
                text_ = "AVG. split: {:s}, epoch [{:d}/{:d}]"
                values = [split, epoch, args.n_epochs]
                for k, v in avg_metrics.items():
                    text_ += ", " + k + ": {:.4f}"
                    values.append(v)
                str_ = text_.format(*values)
                print(datetime.datetime.now())
                print(str_)

            # log to tensorboard at the end of one epoch
            if args.tensorboard:
                # 1. Log scalar values (scalar summary)
                for k, v in metrics_epoch.items():
                    logger.add_scalar('{}/{}'.format(split, k), np.mean(v), epoch)

        # monitor best loss value for early stopping
        # if the early stopping metric is recall (the higher the better),
        # multiply this value by -1 to save the model if the recall increases.
        if args.es_metric.startswith('recall'):
            mult = -1
        else:
            mult = 1

        curr_loss = np.mean(metrics_epoch[args.es_metric])

        if args.lr_decay_factor != -1:
            if args.scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(curr_loss)
            else:
                scheduler.step()

        if curr_loss*mult < best_loss:
            if not args.tensorboard:
                print("Updating best checkpoint")
            save_model(model, optimizer, 'best', checkpoints_dir, ngpus)
            best_loss = curr_loss*mult

            wait = 0
        else:
            wait += 1

        # save current model state to be able to resume it
        save_model(model, optimizer, 'curr', checkpoints_dir, ngpus)
        args.best_loss = best_loss
        args.curr_epoch = epoch
        pickle.dump(args, open(os.path.join(checkpoints_dir,
                                            'args.pkl'), 'wb'))

        if wait == args.patience:
            break

    if args.tensorboard:
        logger.close()


def main():
    args = get_args()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    train(args)


if __name__ == "__main__":
    main()
