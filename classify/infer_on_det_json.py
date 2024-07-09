# %% set up environment
from builtins import Exception, enumerate
import sys
# sys.path.append("/root/workspace/data/classify")
import random
import types
from typing import List, Sequence, Tuple, Union
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# from classify.dataset import CellDataset

matplotlib.use('Agg')
import os

join = os.path.join
from tqdm import tqdm
import torch
import argparse
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from skimage import measure, morphology

from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.metrics import  ConfusionMatrixMetric
from monai.metrics.rocauc import  ROCAUCMetric
from monai.transforms import (
        AsDiscrete,
        Compose,
        EnsureType,

    )

from models.model_loader import get_default_model, get_dual_stream_vit, get_vit_base, get_vit_large
from metrics.confuse_matrix import MultiConfuseMatrixMetric
from optims.lion_torch import Lion
from dataset.transforms.transforms import train_transforms, val_transforms
from dataset.transforms.transforms import *
from utils.logging_helper import get_default_logger
from utils.plot_utils import plot_confusion_matrix, plot_f1_curve, plot_roc_curve, plot_f1_threshold, plot_pr_curve

# set seeds
torch.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-train', '--train_file_path', type=str, default='.data/Classify/MWC/train_pairs.txt')
parser.add_argument('-val', '--val_file_path', type=str, default='.data/Classify/MWC/train_pairs.txt')

parser.add_argument('-e', '--exp_name', type=str, default='ViT_B_cell_2_cls')
parser.add_argument('-cn', '--class_names', nargs="+", default=["bg", "car", "truck", "bus", "van", "freight_car"])

parser.add_argument('--task_name', type=str, default='classify')
# parser.add_argument('--model_type', type=str, default='vit_l')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--device', nargs="+", default=["cuda:0"])
parser.add_argument('--work_dir', type=str, default='./work_dir')
parser.add_argument('--log_dir', type=str, default='./logs')
parser.add_argument('--num_class', type=int, default=2)
parser.add_argument('--test', action="store_true")


# train
parser.add_argument('--num_epoch', type=int, default=200)
parser.add_argument('--num_worker', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--tolerate_epoch_num', type=int, default=40)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--eval_interval', type=int, default=5)
parser.add_argument('--eva_start_epoch', type=int, default=4)
parser.add_argument('--amp', action="store_true")
parser.add_argument('--evaluate_first', action="store_true")
parser.add_argument('--evaluate_only', action="store_true")
parser.add_argument('--resume', action="store_true")
parser.add_argument('--noise_ratio', type=float, default=0)

# parser.add_argument('--eva_epoch_interval', type=int, default=80)

parser.add_argument('--lora', action="store_true", help="finetune by low rank adaption method")
parser.add_argument('--save_lora_only', action="store_true", help="how to save the parameters in the LoRA part")
# lora
parser.add_argument('--lora_alpha',  type=int, default=16)
parser.add_argument('--lora_r',  type=int, default=16)
parser.add_argument('--lora_dropout',  type=float, default=1e-2)
parser.add_argument('--lora_target_modules',  nargs="+", default=["Attention", "PatchEmbed"], type=str)
parser.add_argument('--lora_target_submodules',  nargs="+", default=["Linear", "Conv2d"], type=str)
parser.add_argument('--lora_weights_path', type=str, default="")
parser.add_argument('--lora_bias', type=str, default="none")

args = parser.parse_args()

logger = get_default_logger(args.exp_name, "classify/logs/train", use_timestamp=True)
logger.info(args)


def load_checkpoint(model, optimizer, checkpoint, is_resume):
    if checkpoint and os.path.exists(checkpoint):
        cp = torch.load(checkpoint)
        if "model" in checkpoint:
            cp = cp["model"]
        elif "state_dict" in checkpoint:
            cp = cp["state_dict"]
        model.load_state_dict(cp, strict=False)
        if is_resume:
            optimizer.load_state_dict(cp["optimizer"])
            restart_epoch = cp["epoch"] + 1
            del cp
            return restart_epoch
        del cp
    return 0

# debug always true

# model setting
num_classes = args.num_class
checkpoint = args.checkpoint

# wrap lora model
save_lora_only = args.save_lora_only
lora_enable = args.lora
lora_bias = args.lora_bias

# train setting
save_interval = args.save_interval
eval_interval = args.eval_interval
num_epochs = args.num_epoch
batch_size = args.batch_size
num_worker = args.num_worker
amp_enable = args.amp
class_names = args.class_names

# noise learning
noise_ratio = args.noise_ratio
do_noise_learning = True if noise_ratio > 0 else False

if save_interval <= 0:
    save_interval = 1e6
if eval_interval <= 0:
    eval_interval = 1e6

# convert to str
class_names = list(map(str, class_names))

assert len(class_names) == num_classes, f"Category number({num_classes}) is not equal to the length class names({class_names})"

# %% setup model for fine-tuning 
device = args.device[0]

model_save_path = join(args.work_dir, args.task_name)
os.makedirs(model_save_path, exist_ok=True)

model = get_dual_stream_vit(num_class=num_classes)
model = model.to(device)

# optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=0,
                                    dampening=0,
                                    weight_decay=args.weight_decay,
                                    nesterov=False)


start_epoch = load_checkpoint(model, optimizer, checkpoint, is_resume=args.resume)

### Setup losses

# seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
cls_loss = torch.nn.CrossEntropyLoss()
# MultiLabelSoftMarginLoss
# regress loss for IoU/DSC prediction; (ignored for simplicity but will definitely included in the near future)
# regress_loss = torch.nn.MSELoss(reduction='mean')
#%% train

losses = []
best_loss = 1e10
best_metric = 0
best_metric_epoch = 0

# train_dict, val_dict = split_train_val(parse_train_txt(args.train_file_path), val_ratio=0.1)

from dataset.ds_datasets import DualStreamDataset
from monai.data.dataloader import DataLoader

cname2id = {cname: idx for idx, cname in enumerate(class_names)}

train_dataset = DualStreamDataset(args.train_file_path, cname2id, transform=train_transforms, with_noise_label=do_noise_learning, noise_ratio=noise_ratio)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True)

val_dataset = DualStreamDataset(args.val_file_path, cname2id, transform=val_transforms)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size * 2, num_workers=num_worker * 2, shuffle=False, drop_last=False)

val_metric_roc_auc = ROCAUCMetric() 
val_metric_cm =ConfusionMatrixMetric(metric_name=["sensitivity", "specificity", "precision", "accuracy"])
val_metric_mcm = MultiConfuseMatrixMetric(argmax=False, threshold=0.5)

post_pred = Compose(
    [EnsureType()]
)
post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=num_classes)])

to_onehot = AsDiscrete(argmax=True, to_onehot=num_classes)

# AMP
grad_scaler = GradScaler(enabled=amp_enable)

from torch.utils.tensorboard import SummaryWriter

tensorboard_logs_path = join(args.log_dir, args.exp_name)
writer = SummaryWriter(tensorboard_logs_path)
global_step = 0
# if evaluate before train, the start_epoch should -1
if args.evaluate_first:
    start_epoch = start_epoch - 1
assert start_epoch <= num_epochs

for epoch in range(start_epoch, num_epochs):
    epoch_loss = 0

    model.train()

    if not args.evaluate_first and not args.evaluate_only:
        train_iter = tqdm(train_dataloader)
        for step, batch_data in enumerate(train_iter, 1):
            image_rgb, image_tir, label_cid = batch_data["rgb"].to(device), batch_data["tir"].to(device), batch_data["label"].to(device)

            optimizer.zero_grad()

            with autocast(amp_enable):
                pred = model(torch.concat([image_rgb, image_tir], dim=1))
            
            label_cid: torch.Tensor
            if label_cid.ndim == 2:
                label_cid = label_cid[:, 0]
            
            loss = cls_loss(pred, label_cid.to(device).long())

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            ### Logging ...
            train_iter.set_postfix_str(f"step: {step}, loss: {loss.item():.3f}")
            train_iter.set_description_str(f"Epoch[{epoch}]")
            epoch_loss += loss.item()
            
            global_step += 1

        
        epoch_loss /= step
        losses.append(epoch_loss)
        writer.add_scalar("loss", epoch_loss, global_step=epoch)
        logger.info(f'Epoch[{epoch}] Loss: {epoch_loss}')
    
        # save the model checkpoint
        if (epoch + 1) % save_interval == 0:
            logger.info(f"Epoch[{epoch}] Saving model ...")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_metrics": best_metric}, join(model_save_path, 'model_latest.pth'))
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_metrics": best_metric}, join(model_save_path, f'model_epoch_{epoch: 05}.pth'))
            logger.info(f"Epoch[{epoch}] Saved model to {model_save_path}")

    eval_stage = epoch >= args.eva_start_epoch and (epoch + 1) % eval_interval == 0
    if eval_stage or args.evaluate_first or args.evaluate_only:
        model.eval()
        val_metric_cm.reset()
        val_metric_roc_auc.reset()
        val_metric_mcm.reset()
        val_data_length = len(val_dataloader)
        print_intrval = max(val_data_length // 100, 1)
        logger.info(f"Epoch[{epoch}] Start evaluation.")
        for step, batch_data in enumerate(val_dataloader):
            image_rgb, image_tir, label_cid = batch_data["rgb"].to(device), batch_data["tir"].to(device), batch_data["label"].to(device)

            with torch.no_grad():
                # val_outputs_logit = sliding_window_inference(image, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=predictor, device=device, mode="gaussian", overlap=0.5)
                raw_pred = model(torch.concat([image_rgb, image_tir], dim=1))
                pred = torch.softmax(raw_pred, dim=1)
                            
            val_outputs = [post_pred(i) for i in decollate_batch(pred)]
            val_outputs_onehot = [to_onehot(post_pred(i)) for i in decollate_batch(pred)]
            val_labels = [post_gt(i).to(device) for i in decollate_batch(label_cid)]
            
            #  metrics
            res = val_metric_roc_auc(val_outputs, val_labels)
            # print(label_cid.device, val_outputs_onehot[0].device, val_labels[0].device)
            res2 = val_metric_cm(val_outputs_onehot, val_labels)
            _ = val_metric_mcm([post_pred(i) for i in decollate_batch(raw_pred)], val_labels)
            
            display_res = str(res[0][0].cpu().numpy())
            # display_res2 = str(res2[0].cpu().numpy())
            # print(batch_data.keys())
            # filename = os.path.basename(
            #     batch_data["img_meta_dict"]["filename_or_obj"][0]
            # )
            if step % print_intrval == 0:
                logger.info(f"Val[{step}/{len(val_dataloader)}]: {display_res}")
                
        auc = val_metric_roc_auc.aggregate(average="macro")
        auc_classwise = val_metric_roc_auc.aggregate(average="none")
        auc_classwise = {class_names[i]: _auc for i, _auc in enumerate(auc_classwise)}
        cm_res = val_metric_cm.aggregate()
        cm = val_metric_mcm.aggregate()

        sensitivity, specificity, precision, accuracy = cm_res[0].item(), cm_res[1].item(), cm_res[2].item(), cm_res[3].item()
        auc_metirc = auc
        
        ### Logging ...
        
        # try:
        cm_image = plot_confusion_matrix(cm.numpy(), nc=num_classes, names=class_names, show=False)
        writer.add_images("confuse_matrix", cm_image, global_step=epoch, dataformats="HWC")
        y_pred, y = val_metric_roc_auc.get_buffer()
        roc_auc_image = plot_roc_curve(y, y_pred, class_names=class_names)
        pr_image = plot_pr_curve(y, y_pred, class_names=class_names)
        f1_image = plot_f1_curve(y, y_pred, class_names=class_names)
        writer.add_images("roc_auc_curve", roc_auc_image, global_step=epoch, dataformats="HWC")
        writer.add_images("pr_mp_curve", pr_image, global_step=epoch, dataformats="HWC")
        writer.add_images("f1_curve", f1_image, global_step=epoch, dataformats="HWC")

        # except Exception as e:
        #     logger.error(e)
        writer.add_scalars("classify metrics", {"auc": auc_metirc, "sensitivity": sensitivity, "precision": specificity, "precision": precision, "accuracy": accuracy}, global_step=epoch)
        
        logger.info(f"Epoch[{epoch}] AUC: {auc_metirc}, Sensitivity: {sensitivity}, Precision: {precision}, Specificity: {specificity}, Accuracy: {accuracy}")
        logger.info(f"ROC-AUC Classwise: {auc_classwise}")
        
        if args.evaluate_only:
            break
        
        # save the best model
        if auc > best_metric:
            best_metric = auc
            best_metric_epoch = epoch
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_metrics": best_metric}, join(model_save_path, 'model_best.pth'))

            
        if epoch - best_metric_epoch > args.tolerate_epoch_num:
            logger.info(f"Stop training after {epoch} epochs, more then {epoch - best_metric_epoch} epochs model has not improve, the best metirc is {best_metric} at epoch {best_metric_epoch}") 
            
            break
        
        args.evaluate_first = False
    
# save final model
if not args.evaluate_only:
    if (lora_enable and  not save_lora_only) or not lora_enable:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_metrics": best_metric}, join(model_save_path, 'model_final.pth'))
        
    plt.plot(losses)
    plt.title(f'Train Loss[{cls_loss._get_name()}]')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.show() # comment this line if you are running on a server
    plt.savefig(join(model_save_path, 'train_loss.png'))
    plt.close()
else:
    pass