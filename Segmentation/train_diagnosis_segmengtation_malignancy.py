from dataset.datasets_task_prompt import load_data_volume
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from utils.script_util import save_checkpoint
import sys
from monai.losses import DiceCELoss, DiceLoss
from modeling.Med_SAM.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
import torch.nn.functional as F
from modeling.Med_SAM.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
from modeling.Med_SAM.mask_decoder import Cls_Decoder
import torch
from modeling.Med_SAM.prompt_encoder import PromptEncoder, TwoWayTransformer
from modeling.Med_SAM.prompt_encoder import TaskPromptEncoder_Malignancy as TaskPromptEncoder
import torch.nn as nn
from functools import partial
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from utils.util import setup_logger
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kidney"]
    )
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=0,
        nargs='+', type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("-tolerance", default=5, type=int)

    # Checkpoints of Step 1
    file = "XXXXXX" 

    args = parser.parse_args()
    device = args.device
    if args.rand_crop_size == 0:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))
    train_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_prefix,
        batch_size=1,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker = args.num_worker
    )

    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        cubic_window_size=8,
        out_chans=256,
        num_slice = 16)
    img_encoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["encoder_dict"], strict=True)

    img_encoder.to(device)
    for p in img_encoder.parameters():
        p.requires_grad = False
    img_encoder.depth_embed.requires_grad = False
    for p in img_encoder.slice_embed.parameters():
        p.requires_grad = False
    for i in img_encoder.blocks:
        for p in i.norm1.parameters():
            p.requires_grad = False
        for p in i.adapter.parameters():
            p.requires_grad = False
        for p in i.norm2.parameters():
            p.requires_grad = False
        i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=False)
    for i in img_encoder.neck_3d:
        for p in i.parameters():
            p.requires_grad = False

    prompt_encoder = PromptEncoder(transformer=TwoWayTransformer(depth=2,
                                                                 embedding_dim=256,
                                                                 mlp_dim=2048,
                                                                 num_heads=8))
    prompt_encoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["feature_dict"][3], strict=True)
    prompt_encoder.to(device)

    mask_decoder = VIT_MLAHead(img_size=96, num_classes=2)
    mask_decoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["decoder_dict"],
                          strict=True)
    mask_decoder.to(device)
  
    Task_prompt_encoder = TaskPromptEncoder()
    Task_prompt_encoder.to(device)

    mask_decoder_2 = Cls_Decoder()
    mask_decoder_2.to(device)
    
    decoder_opt = AdamW([i for i in mask_decoder_2.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)

    feature_opt = AdamW([i for i in Task_prompt_encoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    feature_scheduler = torch.optim.lr_scheduler.LinearLR(feature_opt, start_factor=1.0, end_factor=0.01, total_iters=500)

    loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    patch_size = args.rand_crop_size[0]
    for epoch_num in range(args.max_epoch):
        loss_summary = []
        img_encoder.eval()
        prompt_encoder.eval()
        mask_decoder.eval()
        Task_prompt_encoder.train()
        mask_decoder_2.train()
        for idx, (img, seg, spacing, cls_label) in enumerate(train_data):
          out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
          input_batch = out.to(device)
          input_batch = input_batch[0].transpose(0, 1)
          with torch.no_grad():
            batch_features, feature_list = img_encoder(input_batch)
            feature_list.append(batch_features)
            l = len(torch.where(seg == 1)[0])
            points_torch = None
            if l > 0:
                sample = np.random.choice(np.arange(l), 10, replace=True)
                x = torch.where(seg == 1)[1][sample].unsqueeze(1)
                y = torch.where(seg == 1)[3][sample].unsqueeze(1)
                z = torch.where(seg == 1)[2][sample].unsqueeze(1)
                points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                points_torch = points.to(device)
                points_torch = points_torch.transpose(0,1)
            l = len(torch.where(seg < 10)[0])
            sample = np.random.choice(np.arange(l), 20, replace=True)
            x = torch.where(seg < 10)[1][sample].unsqueeze(1)
            y = torch.where(seg < 10)[3][sample].unsqueeze(1)
            z = torch.where(seg < 10)[2][sample].unsqueeze(1)
            points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
            points_torch_negative = points.to(device)
            points_torch_negative = points_torch_negative.transpose(0, 1)
            if points_torch is not None:
                points_torch = torch.cat([points_torch, points_torch_negative], dim=1)
            else:
                points_torch = points_torch_negative
            
            new_feature = []
            for i, feature in enumerate(feature_list):
                if i == 3:
                    new_feature.append(
                        prompt_encoder(feature, points_torch.clone(), [patch_size, patch_size, patch_size])
                    ) 
                else:
                    new_feature.append(feature)
            img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device), scale_factor=64/patch_size,
                mode='trilinear')
            new_feature.append(img_resize)

            middle_feature = mask_decoder(new_feature, 2, patch_size//64)

          head_pos, head_neg = Task_prompt_encoder(feature_list[3])  
          pos_masks = mask_decoder_2(middle_feature, head_pos, img_resize, patch_size//64)
          neg_masks = mask_decoder_2(middle_feature, head_neg, img_resize, patch_size//64)
          pos_masks = pos_masks.permute(0, 1, 4, 2, 3)
          neg_masks = neg_masks.permute(0, 1, 4, 2, 3)
          seg = seg.to(device)
          seg = seg.unsqueeze(1)
          blank = torch.zeros_like(seg)          
          if cls_label == 1:
               loss1 = loss_cal(pos_masks, seg)
               loss2 = loss_cal(neg_masks, blank)
               loss3 = loss_cal(pos_masks, blank)
          if cls_label == 0:
               loss1 = loss_cal(pos_masks, blank)
               loss2 = loss_cal(neg_masks, seg)
               loss3 = loss_cal(neg_masks, blank)

          loss = (loss1 + loss2)/2.0
          print(cls_label, float(loss1), float(loss2), float(loss3))
          loss_summary.append(loss.detach().cpu().numpy())
          decoder_opt.zero_grad()
          feature_opt.zero_grad()
          loss.backward()
          logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(train_data)) + ": loss:" + str(
                    loss_summary[-1].flatten()[0]))
          torch.nn.utils.clip_grad_norm_(Task_prompt_encoder.parameters(), 1.0)
          torch.nn.utils.clip_grad_norm_(mask_decoder_2.parameters(), 1.0)
          feature_opt.step()
          decoder_opt.step()
        feature_scheduler.step()
        decoder_scheduler.step()

        logger.info("- Train metrics: " + str(np.mean(loss_summary)))

        save_checkpoint({"epoch": epoch_num,
                         "cls_prompt_dict": Task_prompt_encoder.state_dict(),
                         "mask_decoder_2_dict": mask_decoder_2.state_dict(),
                         "feature_opt": feature_opt.state_dict(),
                         "decoder_opt": decoder_opt.state_dict()
                         },
                        is_best=0,
                        checkpoint=args.snapshot_path)

if __name__ == "__main__":
    main()
