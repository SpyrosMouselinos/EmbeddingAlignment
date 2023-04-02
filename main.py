import argparse
import json
import os
import sys
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter
from fromage import data
from fromage import models
from fromage import utils
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
visual_models = ['openai/clip-vit-large-patch14', 'openai/clip-vit-large-patch14-336']
llm_models = ['facebook/opt-125m', 'facebook/opt-350m', 'facebook/opt-1.3b',
              'facebook/opt-2.7b', 'facebook/opt-6.7b']
datasets = ['cc3m']
best_score = 0


def parse_args(args):
    parser = argparse.ArgumentParser(description='TLTL Trainer')
    parser.add_argument('--opt-version', default='facebook/opt-125m')
    parser.add_argument('--inspect', action='store_true', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--visual-model',
                        default='openai/clip-vit-large-patch14',
                        type=str, help="Visual encoder to use.")
    parser.add_argument('-d', '--dataset', metavar='DATASET', default='cc3m')

    parser.add_argument('--val-dataset', metavar='DATASET', default='cc3m')

    parser.add_argument('--dataset_dir', default='./datasets/', type=str,
                        help='Dataset directory containing .tsv files.')

    parser.add_argument('--image-dir', default='./data/', type=str,
                        help='Dataset directory containing image folders.')

    parser.add_argument('--log-base-dir', default='./runs/', type=str,
                        help='Base directory to write logs and ckpts to.')

    parser.add_argument('--exp_name', default='cifar_100', type=str,
                        help='Name of experiment, used for saving checkpoints.')

    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--steps-per-epoch', default=19, type=int, metavar='N',
                        help='number of training steps per epoch')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--val-steps-per-epoch', default=-1, type=int, metavar='N',
                        help='number of validation steps per epoch.')

    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N',
                        help='mini-batch size (default: 180), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--val-batch-size', default=None, type=int)

    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')

    parser.add_argument('--lr-warmup-steps', default=5, type=int,
                        metavar='N', help='Number of steps to warm up lr.')

    parser.add_argument('--lr-schedule-step-size', default=100, type=int,
                        metavar='N', help='Number of epochs between decaying lr.')

    parser.add_argument('--lr-schedule-gamma', default=0.1, type=float,
                        metavar='N', help='Decay parameter for learning rate scheduler.')

    parser.add_argument('--grad-accumulation-steps', default=1, type=int, metavar='N',
                        help='number of gradient accumulation steps')

    parser.add_argument('--grad-clip', default=-1.0, type=float, help='gradient clipping amount')

    parser.add_argument('--precision', default='fp32', type=str, choices=['fp32', 'fp16', 'bf16'],
                        help="Precision to train in.")

    parser.add_argument('--cap-loss-scale', type=float, default=1.0, help="Scale on captioning loss.")
    parser.add_argument('--ret-loss-scale', type=float, default=1.0, help="Scale on retrieval loss.")

    parser.add_argument('--concat-captions-prob', type=float, default=0.5,
                        help="Probability of concatenating two examples sequentially for captioning.")

    parser.add_argument('--concat-for-ret', action='store_true', default=False,
                        help="Whether to concatenate examples for retrieval mode.")

    parser.add_argument('--input-prompt', default=None, type=str, help="Input prompt for the language model, if any.")

    parser.add_argument('--image-size', default=224, type=int, metavar='N', help='Size of images.')

    parser.add_argument('--use_image_embed_norm', action='store_true', default=False,
                        help="Whether to use norm on the image embeddings to make them equal to language.")

    parser.add_argument('--image_embed_dropout_prob', type=float, default=0.0,
                        help="Dropout probability on the image embeddings.")

    parser.add_argument('--use_text_embed_layernorm', action='store_true', default=False,
                        help="Whether to use layer norm on the text embeddings for retrieval.")

    parser.add_argument('--text_embed_dropout_prob', type=float, default=0.0,
                        help="Dropout probability on the text embeddings.")

    parser.add_argument('--shared-emb-dim', default=256, type=int, metavar='N',
                        help='Embedding dimension for retrieval.')

    parser.add_argument('--text-emb-layers', help='Layer to use for text embeddings. OPT-2.7b has 33 layers.',
                        default='-1',
                        type=lambda s: [int(x) for x in s.split(',')])

    parser.add_argument('--max-len', default=24, type=int,
                        metavar='N', help='Maximum length to truncate captions / generations to.')

    parser.add_argument('--n-visual-tokens', default=1, type=int,
                        metavar='N', help='Number of visual tokens to use for the Frozen model.')

    parser.add_argument('--beta1', default=0.9, type=float, metavar='M', help='beta1 for Adam')
    parser.add_argument('--beta2', default=0.95, type=float, metavar='M', help='beta2 for Adam')
    parser.add_argument('--wd', '--weight-decay', default=0.001, type=float,
                        metavar='W', help='weight decay (default: 0.0)', dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=1, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')

    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')

    parser.add_argument('--dist-url', default='tcp://127.0.0.1:1337', type=str,
                        help='url used to set up distributed training')

    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')

    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')

    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    parser.add_argument('--save_images', default='True', type=str,
                        help='weather to save images into a folder while iterating or not')

    parser.add_argument('--multiprocessing-distributed', default=False, action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--gamma_neg', default=1, type=int,
                        help='Scale of negative example impact in assymetric entropy loss')
    parser.add_argument('--gamma_pos', default=1, type=int,
                        help='Scale of positive example impact in assymetric entropy loss')

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    i = 1
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    while os.path.exists(args.log_dir):
        args.log_dir = os.path.join(args.log_base_dir, f'{args.exp_name}_{i}')
        i += 1
    os.makedirs(args.log_dir)

    with open(os.path.join(args.log_dir, f'args.json'), 'w') as wf:
        json.dump(vars(args), wf, indent=4)

    print(f'Logging to {args.log_dir}.')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    """Setup code."""
    global best_score
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Create model
    model_args = models.FrozenArgs()
    model_args.opt_version = args.opt_version
    model_args.freeze_lm = True
    model_args.visual_encoder = args.visual_model
    model_args.freeze_vm = True
    model_args.n_visual_tokens = args.n_visual_tokens
    model_args.use_image_embed_norm = args.use_image_embed_norm
    model_args.image_embed_dropout_prob = args.image_embed_dropout_prob
    model_args.use_text_embed_layernorm = args.use_text_embed_layernorm
    model_args.text_embed_dropout_prob = args.text_embed_dropout_prob
    model_args.shared_emb_dim = args.shared_emb_dim
    model_args.text_emb_layers = args.text_emb_layers

    tokenizer = AutoTokenizer.from_pretrained(args.opt_version, use_fast=False)
    # Add an image token for loss masking (and visualization) purposes.
    tokenizer.add_special_tokens({"cls_token": "<|image|>"})  # add special image token to tokenizer
    print('Adding [RET] token to vocabulary.')
    print('Before adding new token, tokenizer("[RET]") =', tokenizer('[RET]', add_special_tokens=False))
    num_added_tokens = tokenizer.add_tokens('[RET]')
    print(f'After adding {num_added_tokens} new tokens, tokenizer("[RET]") =',
          tokenizer('[RET]', add_special_tokens=False))
    ret_token_idx = tokenizer('[RET]', add_special_tokens=False).input_ids
    assert len(ret_token_idx) == 1, ret_token_idx
    model_args.retrieval_token_idx = ret_token_idx[0]
    args.retrieval_token_idx = ret_token_idx[0]

    # Save model args to disk.
    with open(os.path.join(args.log_dir, 'model_args.json'), 'w') as f:
        json.dump(vars(model_args), f, indent=4)

    model = models.TLTLModel(tokenizer, model_args)
    # Print parameters and count of model.
    param_counts_text = utils.get_params_count_str(model)
    with open(os.path.join(args.log_dir, 'param_count.txt'), 'w') as f:
        f.write(param_counts_text)

    # Log trainable parameters to Tensorboard.
    _, total_trainable_params, total_nontrainable_params = utils.get_params_count(model)
    writer = SummaryWriter(args.log_dir)
    writer.add_scalar('params/total', total_trainable_params + total_nontrainable_params, 0)
    writer.add_scalar('params/total_trainable', total_trainable_params, 0)
    writer.add_scalar('params/total_non_trainable', total_nontrainable_params, 0)
    writer.close()

    optimizer = torch.optim.AdamW(
        params=[
            {'params': model.visual_embeddings.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': model.post_visual_embeddings_normalization.parameters(), 'lr': args.lr, 'weight_decay': 0}
        ],
        betas=(args.beta1, args.beta2),
        eps=1e-8
    )

    scaler = torch.cuda.amp.GradScaler()

    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    scheduler_steplr = StepLR(optimizer,
                              step_size=args.lr_schedule_step_size * args.steps_per_epoch,
                              gamma=args.lr_schedule_gamma)

    scheduler = GradualWarmupScheduler(optimizer,
                                       multiplier=1.0,
                                       total_epoch=args.lr_warmup_steps,
                                       after_scheduler=scheduler_steplr)
    optimizer.zero_grad()
    optimizer.step()

    if not torch.cuda.is_available():
        print('WARNING: using CPU, this will be slow!')
        model = torch.nn.DataParallel(model)
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.val_batch_size = int((args.val_batch_size or args.batch_size) / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                              find_unused_parameters=False)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.exists(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_score']
            if args.gpu is not None:
                # best_score may be from a checkpoint from a different GPU
                best_score = best_score.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code

    # train_dataset = data.get_dataset(args, 'train', tokenizer, max_items=50_000)
    # val_dataset = data.get_dataset(args, 'val', tokenizer, max_items=10_000)
    train_dataset = data.get_cifar100_dataset(args, 'train', tokenizer)
    val_dataset = data.get_cifar100_dataset(args, 'val', tokenizer)
    print(f'Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples.')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if args.inspect:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)

        inspect(val_loader, model, args, tokenizer)
        return
    else:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Train for one epoch
        print(f"Performing Training Epoch: {epoch}")
        train(train_loader, model, optimizer, epoch, scheduler,scaler, args)

        # Evaluate on validation set
        is_best = False
        best_score = -1
        if epoch % 3 == 2:
            eval_loss, eval_accs = evaluate(val_loader, model, args, epoch)

            # remember best score and save checkpoint
            is_best = eval_loss > best_score
            best_score = max(eval_loss, best_score)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_score': best_score,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, os.path.join(args.log_dir, 'ckpt'))


def train(train_loader, model, optimizer, epoch, scheduler, scaler, args):
    """Main training loop."""
    ngpus_per_node = torch.cuda.device_count()
    batch_time = utils.AverageMeter('Time', ':6.3f')
    cap_time = utils.AverageMeter('CaptioningTime', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    top24 = utils.AverageMeter('Acc@24', ':6.2f')

    writer = SummaryWriter(args.log_dir)

    progress = utils.ProgressMeter(
        args.steps_per_epoch,
        [batch_time, losses, top1, top5, top24],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    start = time.time()

    for i, (images, tokens, caption_len, image_index) in enumerate(train_loader):
        actual_step = epoch * args.steps_per_epoch + i + 1
        data_time.update(time.time() - start)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            images = images.cuda(args.gpu, non_blocking=True)

            visual_embs, _, logits, loss = model(images=images,
                                                 labels=tokens,
                                                 tensors=None,
                                                 mode='no_projection')

            acc1, acc5, acc24 = utils.accuracy(logits.detach().cpu().float(),
                                               tokens.detach().cpu().float(),
                                               -100, topk=(1, 5, 24))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            top24.update(acc24[0], images.size(0))

            loss = loss / args.grad_accumulation_steps
            losses.update(loss.item(), images.size(0))
            scaler.scale(loss).backward()

            if ((i + 1) % args.grad_accumulation_steps == 0) or (i == args.steps_per_epoch - 1):
                # compute gradient and do SGD step
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            batch_time.update(time.time() - start)

            if actual_step == 1 or (i + 1) % args.print_freq == 0:
                ex_per_sec = args.batch_size / batch_time.avg
                if args.distributed:
                    batch_time.all_reduce()
                    data_time.all_reduce()
                    ex_per_sec = (args.batch_size / batch_time.avg) * ngpus_per_node

                    losses.all_reduce()
                    top1.all_reduce()
                    top5.all_reduce()
                    cap_time.all_reduce()

                progress.display(i + 1)

                writer.add_scalar('train/loss', losses.avg, actual_step)
                writer.add_scalar('train/seq_top1_acc', top1.avg, actual_step)
                writer.add_scalar('train/seq_top5_acc', top5.avg, actual_step)
                writer.add_scalar('train/seq_top24_acc', top24.avg, actual_step)
                writer.add_scalar('metrics/total_secs_per_batch', batch_time.avg, actual_step)
                writer.add_scalar('metrics/total_secs_captioning', cap_time.avg, actual_step)
                writer.add_scalar('metrics/data_secs_per_batch', data_time.avg, actual_step)
                writer.add_scalar('metrics/examples_per_sec', ex_per_sec, actual_step)

                batch_time.reset()
                cap_time.reset()
                data_time.reset()
                losses.reset()
                top1.reset()
                top5.reset()

            if i == args.steps_per_epoch - 1:
                break

            scheduler.step()
            curr_lr = optimizer.param_groups[0]['lr']
            if (actual_step == 1) or (i + 1) % args.print_freq == 0:
                writer = SummaryWriter(args.log_dir)
                writer.add_scalar('train/lr', curr_lr, actual_step)
                writer.close()

    writer.close()


def evaluate(val_loader, model, args, epoch):
    """Main evaluation loop."""
    losses = utils.AverageMeter('Val Loss', ':.4e')
    top1 = utils.AverageMeter('Val Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Val Acc@5', ':6.2f')
    top24 = utils.AverageMeter('Val Acc@24', ':6.2f')

    writer = SummaryWriter(args.log_dir)
    model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for i, (images, tokens, caption_len, image_index) in enumerate(val_loader):
                if torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)

                visual_embs, _, logits, loss = model(images=images,
                                                     labels=tokens,
                                                     tensors=None,
                                                     mode='no_projection')

                acc1, acc5, acc24 = utils.accuracy(logits.detach().cpu().float(),
                                                   tokens.detach().cpu().float(),
                                                   -100, topk=(1, 5, 24))

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                top24.update(acc24[0], images.size(0))
                losses.update(loss.item(), images.size(0))

    if args.distributed:
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()
        top24.all_reduce()

    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/seq_top1_acc', top1.avg, epoch)
    writer.add_scalar('val/seq_top5_acc', top5.avg, epoch)
    writer.add_scalar('val/seq_top24_acc', top24.avg, epoch)
    writer.close()

    return losses.avg, (top1.avg, top5.avg, top24.avg)


def inspect(val_loader, model, args, tokenizer):
    model.eval()
    with torch.no_grad():
        for i, (images, tokens, caption_len, image_index) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)

            visual_embs, logits, loss = model(images=images,
                                              labels=tokens,
                                              tensors=None,
                                              mode='no_projection')

            acc1, acc5, acc24 = utils.accuracy(logits.detach().cpu().float(),
                                               tokens.detach().cpu().float(),
                                               -100, topk=(1, 5, 24))
            bd = tokenizer.batch_decode(tokens)
            _, pred = logits.topk(24, -1, True, True)
            mbd = tokenizer.batch_decode(pred)
            print(f"Caption: {bd}\n")
            print(f"Model BOW: {mbd}\n")
            print(f"Acc@1:{acc1} /Acc@5:{acc5} / Acc@24:{acc24}")

    return


if __name__ == '__main__':
    main(sys.argv[1:])
