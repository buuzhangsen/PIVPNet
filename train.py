import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
import argparse
from utils.dataset_utils import AdaIRTrainDataset,DerainDehazeDatasetval
from net.model import gUNet
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt

class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = gUNet()
        self.loss_fn = nn.L1Loss()
        self.validation_outputs = []  # 用于收集验证结果

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        # 分别计算三种损失
        l1_loss = self.loss_fn(restored, clean_patch)


        # 加权总损失（根据需求调整权重）
        loss = l1_loss 

        # 记录所有损失（总损失和分项损失）
        self.log("train_loss/total", loss, prog_bar=True, sync_dist=True)

        

        return loss


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=3e-5, weight_decay=1e-4)
        scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer, 
                warmup_epochs=40,
                max_epochs=300
            ),
            "interval": "epoch"
        }
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enhance_path', type=str, default="/project/train/src_repo/AdaIR-main/data/test/low_lightimageenhance/LoLv1/", help='save path of test hazy images')
    parser.add_argument('--derain_path', type=str, default="/project/train/src_repo/AdaIR-main/data/test/derain/Rain100L/", help='save path of test raining images')
    args = parser.parse_args()
    print("Options")
    print(opt)
    # 初始化日志记录器
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="Gunet-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    # 数据集和加载器
    trainset = AdaIRTrainDataset(opt)
    valset = DerainDehazeDatasetval(args)  # 需要实现验证模式
    
    trainloader = DataLoader(
        trainset, 
        batch_size=opt.batch_size, 
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valloader = DataLoader(
        valset,
        # batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    # 修改检查点回调配置部分
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        monitor="val_psnr",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:03d}-{val_psnr:.2f}",
        auto_insert_metric_name=False
    )
    # 新增每个epoch保存的回调
    epoch_checkpoint = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        filename="epoch={epoch:03d}",
        every_n_epochs=1,  # 每个epoch保存一次
        save_top_k=-1,      # 保存所有epoch的检查点
        save_last=False     # 不与"last"检查点冲突
    )
    
    if opt.resume_ckpt:
        print(f"Loading weights from checkpoint: {opt.resume_ckpt}")
        model = AdaIRModel()
        
        # 加载检查点并处理权重
        checkpoint = torch.load(opt.resume_ckpt, map_location='cuda')
        model_dict = model.state_dict()
        
        # 参数匹配逻辑
        pretrained_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if k in model_dict:  # 仅保留匹配的参数
                if v.shape == model_dict[k].shape:  # 检查形状是否一致
                    pretrained_dict[k] = v
                else:
                    print(f"Shape mismatch for {k}: {v.shape} vs {model_dict[k].shape}")
            else:
                print(f"Parameter {k} not found in current model")
        
        # 加载兼容参数
        model_dict.update(pretrained_dict)
        load_info = model.load_state_dict(model_dict, strict=False)
        print(f"Loaded weights: {len(pretrained_dict)}/{len(model_dict)} parameters")
        print("Missing keys:", load_info.missing_keys)
        print("Unexpected keys:", load_info.unexpected_keys)
        
        # 可选：恢复优化器状态
        if opt.resume_optimizer:
            try:
                optimizer_dict = checkpoint['optimizer_states'][0]
                model.configure_optimizers()  # 重新初始化优化器
                model.optimizer.load_state_dict(optimizer_dict)
                print("Optimizer state restored")
            except:
                print("Failed to restore optimizer state")
    else:
        model = AdaIRModel()

    # 训练器配置
    strategy = DDPStrategy(find_unused_parameters=True)
    # 修改训练器配置
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy=strategy,
        logger=logger,
        callbacks=[checkpoint_callback, epoch_checkpoint],  # 添加两个回调
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        resume_from_checkpoint=opt.resume_ckpt if opt.resume_ckpt and opt.strict_resume else None
    )
    
    # 开始训练
    trainer.fit(
        model=model, 
        train_dataloaders=trainloader,
        val_dataloaders=valloader
    )


if __name__ == '__main__':
    main()