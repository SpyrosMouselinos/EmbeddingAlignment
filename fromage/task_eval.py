import collections
from PIL import Image
import time
import tqdm
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import BLEUScore
import torchvision
import os
import sys
import argparse
from fromage.models import TLTLModel, FrozenArgs, LLMGENWrapper
from fromage import utils, data
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')  # !IMPORTANT
os.environ["TOKENIZERS_PARALLELISM"] = "false"
visual_models = ['openai/clip-vit-large-patch14', 'openai/clip-vit-large-patch14-336']
llm_models = ['facebook/opt-125m', 'facebook/opt-350m', 'facebook/opt-1.3b',
              'facebook/opt-2.7b', 'facebook/opt-6.7b']
datasets = ['cc3m']
best_score = 0


def parse_args(args):
    parser = argparse.ArgumentParser(description='TLTL Trainer')
    parser.add_argument('--opt-version', default='facebook/opt-125m')
    parser.add_argument('--visual-model',
                        default='openai/clip-vit-large-patch14',
                        type=str, help="Visual encoder to use.")
    parser.add_argument('-d', '--dataset', metavar='DATASET', default='cc3m')

    parser.add_argument('--val-dataset', metavar='DATASET', default='cc3m')

    parser.add_argument('--dataset_dir', default='../datasets/', type=str,
                        help='Dataset directory containing .tsv files.')

    parser.add_argument('--image-dir', default='../data/', type=str,
                        help='Dataset directory containing image folders.')

    parser.add_argument('--log-base-dir', default='../runs/', type=str,
                        help='Base directory to write logs and ckpts to.')

    parser.add_argument('--exp_name', default='test', type=str,
                        help='Name of experiment, used for saving checkpoints.')

    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--steps-per-epoch', default=196, type=int, metavar='N',
                        help='number of training steps per epoch')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--val-steps-per-epoch', default=-1, type=int, metavar='N',
                        help='number of validation steps per epoch.')

    parser.add_argument('-b', '--batch-size', default=10, type=int,
                        metavar='N',
                        help='mini-batch size (default: 180), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--val-batch-size', default=None, type=int)

    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')

    parser.add_argument('--lr-warmup-steps', default=100, type=int,
                        metavar='N', help='Number of steps to warm up lr.')

    parser.add_argument('--lr-schedule-step-size', default=3, type=int,
                        metavar='N', help='Number of steps before decaying lr.')

    parser.add_argument('--lr-schedule-gamma', default=0.1, type=float,
                        metavar='N', help='Decay parameter for learning rate scheduler.')

    parser.add_argument('--grad-accumulation-steps', default=1, type=int, metavar='N',
                        help='number of gradient accumulation steps')

    parser.add_argument('--grad-clip', default=-1.0, type=float, help='gradient clipping amount')

    parser.add_argument('--precision', default='fp16', type=str, choices=['fp32', 'fp16', 'bf16'],
                        help="Precision to train in.")

    parser.add_argument('--cap-loss-scale', type=float, default=1.0, help="Scale on captioning loss.")
    parser.add_argument('--ret-loss-scale', type=float, default=1.0, help="Scale on retrieval loss.")

    parser.add_argument('--concat-captions-prob', type=float, default=0.5,
                        help="Probability of concatenating two examples sequentially for captioning.")

    parser.add_argument('--concat-for-ret', action='store_true', default=False,
                        help="Whether to concatenate examples for retrieval mode.")

    parser.add_argument('--input-prompt_pre', default=None, type=str,
                        help="Input prompt for the language model, if any.")

    parser.add_argument('--input-prompt_post', default='This is an image of', type=str,
                        help="Input prompt for the language model, if any.")

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

    parser.add_argument('-p', '--print-freq', default=90, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

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
    parser.add_argument('--gamma_neg', default=0, type=int,
                        help='Scale of negative example impact in assymetric entropy loss')
    parser.add_argument('--gamma_pos', default=0, type=int,
                        help='Scale of positive example impact in assymetric entropy loss')

    return parser.parse_args(args)


def run_validate(val_loader, tokenizer, model, lm_model, args, progress):
    with torch.no_grad():
        all_generated_captions = []
        all_gt_captions = []

        for i, (og_img, images, tokens, caption_len, image_index) in tqdm.tqdm(enumerate(val_loader),
                                                                       position=0,
                                                                       total=len(val_loader)):

            if torch.cuda.is_available():
                tokens = tokens.cuda(args.gpu, non_blocking=True)
                caption_len = caption_len.cuda(args.gpu, non_blocking=True)
                images = images.cuda(args.gpu, non_blocking=True)

            visual_embs, _, logits, _ = model(images=images, labels=None, tensors=None)
            emb_list = []
            if args.input_prompt_pre is not None:
                prompt_ids_pre = lm_model.tokenizer(args.input_prompt_pre,
                                                    add_special_tokens=True,
                                                    return_tensors="pt").input_ids

                prompt_ids_pre = prompt_ids_pre.to(visual_embs.device)
                prompt_embs_pre = lm_model.input_embeddings(prompt_ids_pre)
                prompt_embs_pre = prompt_embs_pre.repeat(visual_embs.shape[0], 1, 1)
                emb_list.append(prompt_embs_pre)
            #####
            emb_list.append(visual_embs.unsqueeze(1))
            #####
            if args.input_prompt_post is not None:
                prompt_ids_post = lm_model.tokenizer(args.input_prompt_post,
                                                     add_special_tokens=args.input_prompt_pre is None,
                                                     return_tensors="pt").input_ids

                prompt_ids_post = prompt_ids_post.to(visual_embs.device)
                prompt_embs_post = lm_model.input_embeddings(prompt_ids_post)
                prompt_embs_post = prompt_embs_post.repeat(visual_embs.shape[0], 1, 1)
                emb_list.append(prompt_embs_post)

            input_embs = torch.cat(emb_list, dim=1)

            ##########################################################################################################

            generated_ids, _ = lm_model.generate(input_embs, 20, temperature=0.0, top_p=1)

            newline_token_id = lm_model.tokenizer('\n', add_special_tokens=False).input_ids[0]
            trunc_idx = 0
            for j in range(generated_ids.shape[1]):
                if generated_ids[0, j] == newline_token_id:
                    trunc_idx = j
                    break
            if trunc_idx > 0:
                generated_ids = generated_ids[:, :trunc_idx]

            return_outputs = []
            return_outputs_trunc = []
            caption = lm_model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return_outputs.append(caption)
            print(return_outputs)
            plt.figure(figsize=(12,12))
            ToPILImage()(og_img[0, :]).show()
            plt.close()
            if i % args.print_freq == 0:
                progress.display(i + 1)

            if i == args.val_steps_per_epoch - 1:
                break

        # Measure captioning metrics.
        # path2captions = collections.defaultdict(list)
        # for image_path, caption in zip(all_generated_image_paths, all_gt_captions):
        #     assert len(caption) == 1, caption
        #     path2captions[image_path].append(caption[0].replace('[RET]', ''))
        # full_gt_captions = [path2captions[path] for path in all_generated_image_paths]
        #
        # print(f'Computing BLEU with {len(all_generated_captions)} generated captions:'
        #       f'{all_generated_captions[:5]} and {len(full_gt_captions)} ground truth captions:',
        #       f'{full_gt_captions[:5]}.')
        # bleu1_score = bleu_scorers[0](all_generated_captions, full_gt_captions)
        # bleu1.update(bleu1_score, 1)
        # bleu2_score = bleu_scorers[1](all_generated_captions, full_gt_captions)
        # bleu2.update(bleu2_score, 1)
        # bleu3_score = bleu_scorers[2](all_generated_captions, full_gt_captions)
        # bleu3.update(bleu3_score, 2)
        # bleu4_score = bleu_scorers[3](all_generated_captions, full_gt_captions)
        # bleu4.update(bleu4_score, 3)
        # loss = args.ret_loss_scale * (image_loss + caption_loss) / 2.0


def captioning_evaluation(step, val_loader, model, lm_model, tokenizer, args):
    ngpus_per_node = torch.cuda.device_count()
    writer = SummaryWriter(args.log_dir)
    bleu_scorers = [BLEUScore(n_gram=i) for i in [1, 2, 3, 4]]
    model_modes = 'captioning'

    bleu1 = utils.AverageMeter('BLEU@1', ':6.2f', utils.Summary.AVERAGE)
    bleu2 = utils.AverageMeter('BLEU@2', ':6.2f', utils.Summary.AVERAGE)
    bleu3 = utils.AverageMeter('BLEU@3', ':6.2f', utils.Summary.AVERAGE)
    bleu4 = utils.AverageMeter('BLEU@4', ':6.2f', utils.Summary.AVERAGE)
    top1_caption = utils.AverageMeter('CaptionAcc@1', ':6.2f', utils.Summary.AVERAGE)
    top5_caption = utils.AverageMeter('CaptionAcc@5', ':6.2f', utils.Summary.AVERAGE)
    top24_caption = utils.AverageMeter('CaptionAcc@24', ':6.2f', utils.Summary.AVERAGE)

    progress = utils.ProgressMeter(
        len(val_loader) + (False and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [bleu1, bleu2, bleu3, bleu4, top1_caption, top5_caption, top24_caption],
        prefix='Test: ')

    run_validate(val_loader, tokenizer, model, lm_model, args, progress)

    progress.display_summary()

    writer.add_scalar('test/bleu1', bleu1.avg, step)
    writer.add_scalar('test/bleu2', bleu2.avg, step)
    writer.add_scalar('test/bleu3', bleu3.avg, step)
    writer.add_scalar('test/bleu4', bleu4.avg, step)

    writer.add_scalar('test/t2i_top1_acc', top1_caption.avg, step)
    writer.add_scalar('test/t2i_top5_acc', top5_caption.avg, step)
    writer.add_scalar('test/t2i_top5_acc', top24_caption.avg, step)

    writer.close()

    return (bleu1.avg, bleu2.avg, bleu3.avg, bleu4.avg), (top1_caption.avg, top5_caption.avg, top24_caption.avg)


def main(args):
    args = parse_args(args)
    i = 1
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    while os.path.exists(args.log_dir):
        args.log_dir = os.path.join(args.log_base_dir, f'{args.exp_name}_{i}')
        i += 1
    os.makedirs(args.log_dir)
    # Create model
    model_args = FrozenArgs()
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

    model = TLTLModel(tokenizer, model_args)
    checkpoint = torch.load('../runs/full_model_fp32_assymetric_pos0_neg0/ckpt_best.pth.tar')
    new_checkpoint = {'state_dict': {}}
    for param_name, value in checkpoint['state_dict'].items():
        new_name = '.'.join(param_name.split('.')[1:])
        new_checkpoint['state_dict'].update({new_name: value})
    model.load_state_dict(new_checkpoint['state_dict'])
    model.cuda()
    model.custom_eval()

    lm_model = AutoModelForCausalLM.from_pretrained(args.opt_version)
    wrapped_lm_model = LLMGENWrapper(lm_model, tokenizer)
    wrapped_lm_model.cuda()
    wrapped_lm_model.eval()
    test_dataset = data.get_cifar100_dataset(args, 'val', tokenizer, return_og_image=True)
    print(f'Testing with {len(test_dataset)} examples.')

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

    captioning_evaluation(step=0, val_loader=test_loader, model=model, lm_model=wrapped_lm_model, tokenizer=tokenizer,
                          args=args)


if __name__ == '__main__':
    main(sys.argv[1:])
