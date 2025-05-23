import torch.optim as optim
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from dataset.datasets_diagnosis import load_data_volume
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from model import ClassNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from utils.script_util import save_checkpoint
from util_c import get_roc_auc
from calculate_precision import calc_conf_mat, calc_eval_metric
import sys
from modeling.Med_SAM.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
import torch.nn.functional as F
from modeling.Med_SAM.mask_decoder import Cls_Decoder
from modeling.Med_SAM.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
import torch
from modeling.Med_SAM.prompt_encoder import PromptEncoder, TwoWayTransformer
from modeling.Med_SAM.prompt_encoder import TaskPromptEncoder_Malignancy_s2 as ClsPromptEncoder
from modeling.Med_SAM.prompt_encoder import TaskPromptEncoder_Subtype as ClsPromptEncoder_2
import torch.nn as nn
from functools import partial
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from utils.util import setup_logger
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import surface_distance
from surface_distance import metrics
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
        "--snapshot_path_2",
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
    parser.add_argument(
        "--num_prompts",
        default=10,
        type=int,
    )
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=5, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("-tolerance", default=5, type=int)
    parser.add_argument('--save_interval', type=int, default=10)

    # Checkpoints of Step 1
    file = "XXXXXX" 
    target_class = 6

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
    if not os.path.exists(args.snapshot_path_2):
        os.makedirs(args.snapshot_path_2)
    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))

    train_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_prefix,
        batch_size=args.batch_size,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker=2,
        target_class = target_class
    )
    test_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_prefix,
        batch_size=1,
        augmentation=False,
        split="test",
        rand_crop_spatial_size=args.rand_crop_size,
        convert_to_sam=False,
        do_test_crop=False,
        deterministic=True,
        num_worker=2,
        target_class = target_class
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
    
    prompt_encoder_list = []
    for i in range(4):
        prompt_encoder = PromptEncoder(transformer=TwoWayTransformer(depth=2,
                                                                 embedding_dim=256,
                                                                 mlp_dim=2048,
                                                                 num_heads=8))
        prompt_encoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["feature_dict"][i], strict=True)
        prompt_encoder.to(device)
        prompt_encoder_list.append(prompt_encoder)

    mask_decoder = VIT_MLAHead(img_size=96)
    mask_decoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["decoder_dict"],
                          strict=True)
    mask_decoder.to(device)

    # Checkpoints of Step 2 Malignancy
    snapshot_path_2 = 'XXXXXX'
    cls_prompt_encoder = ClsPromptEncoder()
    cls_prompt_encoder.load_state_dict(torch.load(snapshot_path_2, map_location='cpu')["cls_prompt_dict"], strict=True)
    cls_prompt_encoder.to(device)
    cls_prompt_encoder.eval()

    # Checkpoints of Step 2 Subtype
    snapshot_path_3 = 'XXXXXX'
    cls_prompt_encoder_2 = ClsPromptEncoder_2()
    cls_prompt_encoder_2.load_state_dict(torch.load(snapshot_path_3, map_location='cpu')["cls_prompt_dict"], strict=True)
    cls_prompt_encoder_2.to(device)
    cls_prompt_encoder_2.eval()

    cls_model = ClassNet(in_channels=640+128, num_classes=1)
    cls_model.to(device)

    ce_loss = CrossEntropyLoss()
    optimizer_cls = Adam(cls_model.parameters(), lr=args.lr)

    patch_size = args.rand_crop_size[0]
    img_encoder.eval()
    for i in prompt_encoder_list:
        i.eval()
    mask_decoder.eval()

    def model_predict(img, prompt, img_encoder, prompt_encoder, mask_decoder, cls_prompt_encoder, cls_prompt_encoder_2):
        out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
        input_batch = out[0].transpose(0, 1)
        batch_features, feature_list = img_encoder(input_batch)
        feature_list.append(batch_features)
        points_torch = prompt.transpose(0, 1)
        new_feature = []
        for i, (feature, feature_decoder) in enumerate(zip(feature_list, prompt_encoder)):
            if i == 3:
                new_feature.append(
                    feature_decoder(feature.to(device), points_torch.clone(), [patch_size, patch_size, patch_size])
                )
            else:
                new_feature.append(feature.to(device))

        img_resize = F.interpolate(img[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device), scale_factor=64/patch_size, mode="trilinear")
        new_feature.append(img_resize)
        middle_feature = mask_decoder(new_feature, 2, patch_size//64)
        head = []
        dict = {0:1, 1:1, 2:1, 3:0, 4:0, 5:1}
        for i in range(target_class):
            if dict[i] == 1:
                 cls_tmp = cls_prompt_encoder_2(feature_list[3], i)
                 bin_tmp = cls_prompt_encoder(feature_list[3], pos_embed = cls_prompt_encoder.cls_embedding_pos.flatten(2).permute(0, 2, 1))
            else:
                 cls_tmp = cls_prompt_encoder_2(feature_list[3], i)
                 bin_tmp = cls_prompt_encoder(feature_list[3], neg_embed = cls_prompt_encoder.cls_embedding_neg.flatten(2).permute(0, 2, 1))

            tmp = torch.cat([cls_tmp, bin_tmp], dim=1)
            head.append(tmp)
        return middle_feature, head

    train_losses = []
    eval_losses = []
    aucs = []
    epochs = []

    for epoch in range(args.max_epoch):    
        cls_model.train()
        loss_avg = []
        for idx, (img, seg, flag, label) in enumerate(train_data):
            seg = seg.float()
            prompt = F.interpolate(seg[None, :, :, :, :], img.shape[2:], mode="nearest")[0]
            seg = seg.to(device).unsqueeze(0)
            img = img.to(device)
            label = label.to(device)
            l = len(torch.where(prompt == 1)[0])
            if l < 1:
                continue

            sample = np.random.choice(np.arange(l), args.num_prompts, replace=True)
            x = torch.where(prompt == 1)[1][sample].unsqueeze(1)
            y = torch.where(prompt == 1)[3][sample].unsqueeze(1)
            z = torch.where(prompt == 1)[2][sample].unsqueeze(1)

            x_m = (torch.max(x) + torch.min(x)) // 2
            y_m = (torch.max(y) + torch.min(y)) // 2
            z_m = (torch.max(z) + torch.min(z)) // 2

            d_min = x_m - patch_size//2
            d_max = x_m + patch_size//2
            h_min = z_m - patch_size//2
            h_max = z_m + patch_size//2
            w_min = y_m - patch_size//2
            w_max = y_m + patch_size//2
            d_l = max(0, -d_min)
            d_r = max(0, d_max - prompt.shape[1])
            h_l = max(0, -h_min)
            h_r = max(0, h_max - prompt.shape[2])
            w_l = max(0, -w_min)
            w_r = max(0, w_max - prompt.shape[3])

            d_min = max(0, d_min)
            h_min = max(0, h_min)
            w_min = max(0, w_min)
            d_max = min(d_max, prompt.shape[1])
            h_max = min(h_max, prompt.shape[2])
            w_max = min(w_max, prompt.shape[3])
            
            points = torch.cat([x-d_min, y-w_min, z-h_min], dim=1).unsqueeze(1).float()
            points_torch = points.to(device)
            img_patch = img[:, :,  d_min:d_max, h_min:h_max, w_min:w_max].clone()
            seg_patch = seg[:, :,  d_min:d_max, h_min:h_max, w_min:w_max].clone()
            img_patch = F.pad(img_patch, (w_l, w_r, h_l, h_r, d_l, d_r))
            seg_patch = F.pad(seg_patch, (w_l, w_r, h_l, h_r, d_l, d_r))

            with torch.no_grad():
                middle_feature, heads = model_predict(img_patch,
                                    points_torch,
                                    img_encoder,
                                    prompt_encoder_list,
                                    mask_decoder,
                                    cls_prompt_encoder, 
                                    cls_prompt_encoder_2
                                    )
                seg_resize = F.interpolate(seg_patch[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device), scale_factor=middle_feature.shape[2]/patch_size, mode="trilinear")                
                area = torch.sum(seg_resize.view(seg_resize.size(0), -1), dim=1, keepdim=True)
                pred_labels = None
            
            for i in range(target_class):
                feature_tmp = torch.cat([middle_feature, heads[i]], dim=1)
                feature_tmp = torch.mul(feature_tmp, seg_resize)
                feature_tmp = torch.sum(feature_tmp.view(feature_tmp.size(0), feature_tmp.size(1), -1), axis=2)
                feature_tmp = feature_tmp / area
                pred_label = cls_model(feature_tmp)
                if pred_labels == None:
                    pred_labels = pred_label
                else:
                    pred_labels = torch.cat([pred_labels, pred_label], dim=1)

            loss_cls_label = ce_loss(pred_labels, label)
            
            optimizer_cls.zero_grad()
            loss_cls_label.backward()
            optimizer_cls.step()

            losses = []
            losses.append(loss_cls_label.item())
            loss_avg.append(losses)

        loss_avg = np.mean(np.array(loss_avg), axis=0)
        train_losses.append(loss_avg)
        epochs.append(epoch)
        print('loss train:', epoch, loss_avg[0])

        cls_model.eval()
        loss_avg = []
        output_scores = np.array([]).reshape(0, target_class)
        true_labels = np.array([])
        for idx, (img, seg, flag, label) in enumerate(test_data):
            seg = seg.float()
            prompt = F.interpolate(seg[None, :, :, :, :], img.shape[2:], mode="nearest")[0]
            seg = seg.to(device).unsqueeze(0)
            img = img.to(device)

            label = label.to(device)
            l = len(torch.where(prompt == 1)[0])
            if l < 1:
                continue

            sample = [0, int(l/9), int(l*2/9), int(l*3/9), int(l*4/9), int(l*5/9), int(l*6/9), int(l*7/9), int(l*8/9), l-1]
            x = torch.where(prompt == 1)[1][sample].unsqueeze(1)
            y = torch.where(prompt == 1)[3][sample].unsqueeze(1)
            z = torch.where(prompt == 1)[2][sample].unsqueeze(1)

            x_m = (torch.max(x) + torch.min(x)) // 2
            y_m = (torch.max(y) + torch.min(y)) // 2
            z_m = (torch.max(z) + torch.min(z)) // 2

            d_min = x_m - patch_size//2
            d_max = x_m + patch_size//2
            h_min = z_m - patch_size//2
            h_max = z_m + patch_size//2
            w_min = y_m - patch_size//2
            w_max = y_m + patch_size//2
            d_l = max(0, -d_min)
            d_r = max(0, d_max - prompt.shape[1])
            h_l = max(0, -h_min)
            h_r = max(0, h_max - prompt.shape[2])
            w_l = max(0, -w_min)
            w_r = max(0, w_max - prompt.shape[3])

            d_min = max(0, d_min)
            h_min = max(0, h_min)
            w_min = max(0, w_min)
            d_max = min(d_max, prompt.shape[1])
            h_max = min(h_max, prompt.shape[2])
            w_max = min(w_max, prompt.shape[3])
            
            points = torch.cat([x-d_min, y-w_min, z-h_min], dim=1).unsqueeze(1).float()
            points_torch = points.to(device)
            img_patch = img[:, :,  d_min:d_max, h_min:h_max, w_min:w_max].clone()
            seg_patch = seg[:, :,  d_min:d_max, h_min:h_max, w_min:w_max].clone()
            img_patch = F.pad(img_patch, (w_l, w_r, h_l, h_r, d_l, d_r))
            seg_patch = F.pad(seg_patch, (w_l, w_r, h_l, h_r, d_l, d_r))

            with torch.no_grad():
                middle_feature, heads = model_predict(img_patch,
                                    points_torch,
                                    img_encoder,
                                    prompt_encoder_list,
                                    mask_decoder,
                                    cls_prompt_encoder, 
                                    cls_prompt_encoder_2
                                    )
                seg_resize = F.interpolate(seg_patch[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device), scale_factor=middle_feature.shape[2]/patch_size, mode="trilinear")                
                area = torch.sum(seg_resize.view(seg_resize.size(0), -1), dim=1, keepdim=True)
                pred_labels = None
                for i in range(target_class):
                    feature_tmp = torch.cat([middle_feature, heads[i]], dim=1)
                    feature_tmp = torch.mul(feature_tmp, seg_resize)
                    feature_tmp = torch.sum(feature_tmp.view(feature_tmp.size(0), feature_tmp.size(1), -1), axis=2)
                    feature_tmp = feature_tmp / area
                    pred_label = cls_model(feature_tmp)
                    if pred_labels == None:
                        pred_labels = pred_label
                    else:
                        pred_labels = torch.cat([pred_labels, pred_label], dim=1)
                output = softmax(pred_labels, dim=1)    
            output_scores = np.append(output_scores, output.cpu().numpy(), axis=0)
            true_labels = np.append(true_labels, label.cpu().numpy())

            losses = []
            losses.append(loss_cls_label.item())
            loss_avg.append(losses)

        loss_avg = np.mean(np.array(loss_avg), axis=0)
        auc = get_roc_auc(true_labels, output_scores)
        conf_mat = calc_conf_mat(true_labels, output_scores)
        sen, spe, acc = calc_eval_metric(conf_mat)
        print('loss vl:', epoch, loss_avg[0], 'acc:', acc[-1], 'auc:', auc[-1], "sen:", sen[0], "spe:", spe[0])
        print(conf_mat)
        print(auc)
        eval_losses.append(loss_avg)
        aucs.append(auc[-1])

        # Plotting Loss
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses, 'r', label='Training loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, eval_losses, 'r', label='Eval loss')
        plt.title('Eval Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting Accuracy
        plt.subplot(1, 3, 3)
        plt.plot(epochs, aucs, 'b', label='AUC')
        plt.title('AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()

        # Save the plot as an image file
        plt.savefig(args.snapshot_path_2+'/training_results.png')

        with open(args.snapshot_path_2+'result.txt', 'a') as file:
            np.savetxt(file, [epoch, auc[-1], acc[-1]], fmt='%f', delimiter=',')
            np.savetxt(file, conf_mat, fmt='%d', delimiter=',')

        save_checkpoint({"epoch": epoch,
                         "cls_model_dict": cls_model.state_dict(),
                         },
                        is_best=0,
                        checkpoint=args.snapshot_path_2)

    print('done')        
            
if __name__ == "__main__":
    main()

