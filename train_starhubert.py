import os
import re
import yaml
import random
import argparse
import logging
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from s3prl.optimizers import get_optimizer

from utils import *
from models import init_model

from importlib import reload
logging.shutdown()
reload(logging)

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.strategies import DDPStrategy

import wandb
from pytorch_lightning.loggers import WandbLogger

class W2V2Distil(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.yaml_cfg = cfg
        self.train_cfg = cfg['train']

        # Load teacher model
        teacher_model = self.yaml_cfg['teacher']['teacher_model']
        teacher_cfg = self.yaml_cfg['teacher']
        if 'wavlm' in teacher_model:
            self.teacher_model, teacher_config, self.task_agnostic = load_wavlm_and_config(teacher_model, arg_overrides=teacher_cfg)
        else:
            self.teacher_model, teacher_config, self.task_agnostic = load_model_and_config(teacher_model, arg_overrides=teacher_cfg)
        freeze_model(self.teacher_model)

        # Make student config independent of teacher
        self.distiller_cfg = self.yaml_cfg['distiller']
        init_student_config, init_student_model = init_model(self.yaml_cfg['model'])
        student_config = init_student_config(**self.distiller_cfg)
        student_config._teacher_task_agnostic = self.task_agnostic

        self.time_tag = get_time_tag()
        dump_yaml(student_config, self.yaml_cfg, self.time_tag)

        # Model Initialize -> Distillation training -> Add FC/Dropout & Fine-tuning
        self.student_model = init_student_model(
            cfg=student_config,
            teacher_model=self.teacher_model
        )

        self.fsp_type = self.train_cfg['fsp_type'] # 'layer', 'intra', ''both
        self.fsp_axis = self.train_cfg['fsp_axis'] # 'channel' or 'time', if channel, project on channel axis
        self.zeroth_input = self.train_cfg['zeroth_input'] # if 'pre_trf', use pretrained feature extractor features
        self.fsp_loss_type = self.train_cfg['fsp_loss_type'] # 'mse' or 'l1'

        if self.train_cfg['specaug']:
            from utils.specaug import SpecAug
            specaug = SpecAug(**self.yaml_cfg['specaug'])
            self.student_model.add_specaug(specaug)

        self.batch_size = self.train_cfg['batch_size']
        self.num_gpus = self.train_cfg['gpus']
        if isinstance(self.num_gpus, list):
            self.num_gpus = len(self.num_gpus)
        data_cfg = self.yaml_cfg['data']
        bucketing_path = data_cfg['bucketing_path']
        libri_root = data_cfg['libri_root']
        train_set = data_cfg['train_set']
        test_set = data_cfg['test_set']

        # download & prepare data
        self.train_data = LibriDataset(
            batch_size=self.batch_size,
            file_path=bucketing_path,
            sets=train_set,
            libri_root=libri_root,
        )
        self.eval_data = LibriDataset(
            batch_size=self.batch_size,
            file_path=bucketing_path,
            sets=['dev-clean'],
            libri_root=libri_root,
        )
        self.test_data = LibriDataset(
            batch_size=self.batch_size,
            file_path=bucketing_path,
            sets=test_set,
            libri_root=libri_root,
        )

        # For better pytorch lightning logging
        logging.shutdown()
        reload(logging)

    def forward(self, x, padding_mask=None):
        # TODO: adapt model input and return to StarHuBERT!!!
        # Seems like lightning had been using the teacher model as training mode the whole time
        self.teacher_model.eval()

        teacher_results = self.teacher_model.extract_features(
            source=x, 
            padding_mask=padding_mask,
            mask=False,
        )
        # -> RETURNS: {
        #     "x": (B x T x D) (encoder output),
        #     "layer_results": [x, (attn, lr)] x #layers, 
        #     "features": [features]
        # }

        student_results = self.student_model(
            source=x, 
            padding_mask=padding_mask,
            layer=None, # layer is to break at specific transformer layer and out put that layer results
        )
        # -> RETURNS: {
        #     "x": x, 
        #     "post_cnn": features_to_distill, 
        #     "pre_trf": tr_layer_results, # the input of the first transformer layer (different with feature extractor out put, because there are positional coding and mask)
        #     "layer_results": layer_results, # the output of each transformer layer
        #     "attn_layer_results": attn_layer_results, # attention maps of each layer
        #     "padding_mask": padding_mask,
        # }

        return student_results, teacher_results

    def training_step(self, batch, batch_idx):
        student_results, teacher_results = self(**batch)
        
        loss, losses = self.calculate_loss(student_results, teacher_results)

        if self.train_cfg['monitor_losses']:
            for k, v in losses.items():
                self.log(k, v.item(), prog_bar=True)

        return loss

    def training_epoch_end(self, training_step_outputs):
        pass
    
    def validation_step(self, batch, batch_idx):
        student_results, teacher_results = self(**batch)
        total_loss, losses = self.calculate_loss(student_results, teacher_results)
        self.log("v_loss", total_loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.log("val_layer_loss", losses["layer_loss"], on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.log("val_intra_loss", losses["intra_loss"], on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

        return {"v_loss": total_loss}
    
    def test_step(self, batch, batch_idx):
        student_results, teacher_results = self(**batch)
        total_loss, losses = self.calculate_loss(student_results, teacher_results)
        self.log("test_loss", total_loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.log("test_layer_loss", losses["layer_loss"], on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.log("test_intra_loss", losses["intra_loss"], on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

        return {"test_loss": total_loss}
    
    def calculate_loss(self, student_results, teacher_results, labels=None):
        losses = {"layer_loss": None, "intra_loss": None}
        total_loss = 0.0
        cal_layer = False
        cal_intra = False
        
        if self.fsp_type == 'layer':
            cal_layer = True
        elif self.fsp_type == 'intra':
            cal_intra = True
        elif self.fsp_type == 'both':
            cal_layer = True
            cal_intra = True
        
        loss_func = None
        if self.fsp_loss_type == 'mse':
            loss_func = F.mse_loss
        elif self.fsp_loss_type == 'l1':
            loss_func = F.l1_loss
        elif self.fsp_loss_type == 'KL':
            loss_func = torch.nn.KLDivLoss(reduction="batchmean")
        
        layer_results_teacher = teacher_results["features"]
        layer_results_teacher.extend([
            teacher_results["layer_results"][l][0].transpose(0, 1) # [B, T, D]
            for l in range(len(teacher_results["layer_results"]))
        ])
        layer_results_teacher = torch.stack(layer_results_teacher, dim=1)

        layer_results_student = [student_results["post_cnn"].transpose(0, 1)] # [B, T, D]
        layer_results_student.extend([
            student_results["layer_results"][l].transpose(0, 1)
            for l in range(len(student_results["layer_results"]))
        ]) # [L + 1, B, T, D]
        layer_results_student = torch.stack(layer_results_student, dim=1) # [B, L + 1, T, D]

        if cal_layer:
            gram_teacher = torch.einsum('bltd,blfd->bltf', layer_results_teacher, layer_results_teacher)
            gram_student = torch.einsum('bltd,blfd->bltf', layer_results_student, layer_results_student)
            if self.fsp_loss_type == "KL":
                t = F.log_softmax(gram_student / torch.sqrt(torch.tensor(layer_results_student.shape[-1])), dim=-1)
                layer_loss = loss_func(
                    F.log_softmax(gram_student / torch.sqrt(torch.tensor(layer_results_student.shape[-1])), dim=-1),
                    F.softmax(gram_teacher / torch.sqrt(torch.tensor(layer_results_teacher.shape[-1])), dim=-1)
                )
            else:
                layer_loss = loss_func(gram_teacher, gram_student)
            losses['layer_loss'] = layer_loss
            total_loss += layer_loss

        if cal_intra:
            layer_results_teacher_0 = layer_results_teacher[:, :-1]
            layer_results_teacher_1 = layer_results_teacher[:, 1:]
            gram_teacher_intra = torch.einsum('bltd,blfd->bltf', layer_results_teacher_0, layer_results_teacher_1)
            layer_results_student_0 = layer_results_student[:, :-1]
            layer_results_student_1 = layer_results_student[:, 1:]
            gram_student_intra = torch.einsum('bltd,blfd->bltf', layer_results_student_0, layer_results_student_1)
            if self.fsp_loss_type == "KL":
                intra_loss = loss_func(
                    F.log_softmax(gram_student_intra / torch.sqrt(torch.tensor(layer_results_student.shape[-1])), dim=-1), 
                    F.softmax(gram_teacher_intra / torch.sqrt(torch.tensor(layer_results_teacher.shape[-1])), dim=-1)
                )
            else:
                intra_loss = loss_func(gram_teacher_intra, gram_student_intra)
            losses['intra_loss'] = intra_loss
            total_loss += intra_loss
        
        return total_loss, losses

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=eval(self.yaml_cfg['optimizer']['lr']))
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.1, verbose=True)
        
        train_batches = len(self.train_dataloader()) // self.num_gpus
        num_training_steps = (self.train_cfg['num_epochs'] * train_batches) // self.train_cfg['accumulate_grad_batches']
        num_warmup_steps = int(num_training_steps * self.yaml_cfg['optimizer']['warmup_proportion'])

        return {
            "optimizer": get_optimizer(
                [self.student_model],
                num_training_steps,
                self.yaml_cfg['optimizer']
            )
        }

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=1,
                          shuffle=True,
                          collate_fn=self.train_data.collate_fn,
                          num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.eval_data,
                          batch_size=1,
                          collate_fn=self.eval_data.collate_fn,
                          num_workers=16)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=1,
                          collate_fn=self.test_data.collate_fn,
                          num_workers=16)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '-cfg', '--config', 
                        help='yaml config path for training')

    parser.add_argument('-m', '--model', default='armhubert',
                        help='define model name')

    parser.add_argument('-t', '--test',
                        action='store_true', help='Enable testing mode')

    args = parser.parse_args()

    YAML_PATH = args.config or './conf/armhubert/armhubert-960.yaml'
    with open(YAML_PATH) as f:
        YAML_CFG = yaml.load(f, Loader = yaml.FullLoader)

    YAML_CFG['model'] = args.model

    batch_size = YAML_CFG['train']['batch_size']
    output_dir = YAML_CFG['train']['base_dir'] + 'results/pretrain/' + YAML_CFG['train']['output_dir']
    checkpoint = YAML_CFG['train']['checkpoint']
    gpus = YAML_CFG['train']['gpus']
    num_epochs = YAML_CFG['train']['num_epochs']
    use_fp16 = 16 if YAML_CFG['train']['use_fp16'] else 32
    use_apex = 'apex' if YAML_CFG['train']['use_apex'] else 'native'
    accumulate_grad_batches = YAML_CFG['train']['accumulate_grad_batches']

    model = W2V2Distil(cfg = YAML_CFG)
    wandb_logger = WandbLogger(project = 'ARMHuBERT',
                               name = model.time_tag,
                               resume = False,
                               sync_tensorboard = True)


    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='checkpoint-{epoch:02d}',
        verbose=True,
        save_last=True,
        save_top_k=3,
        monitor='v_loss',
        mode='min'
    )

    # early_stopping = EarlyStopping(
    #     monitor='v_loss',
    #     patience=15,
    #     verbose=True,
    #     mode='min'
    # )

    trainer = Trainer(
        accelerator = 'gpu',
        devices = 1 if args.test else -1,
        strategy= DDPStrategy(find_unused_parameters=False),
        amp_backend=use_apex,
        #amp_backend = "apex",
        #amp_level = "O2",
        precision=use_fp16, 
        max_epochs=num_epochs,
        sync_batchnorm=True,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=checkpoint_callback,  # [early_stopping, checkpoint_callback]
        logger = wandb_logger,
    )

    if args.test:
        trainer.test(model)
    else:
        trainer.fit(
            model, 
            ckpt_path=os.path.join(output_dir, checkpoint) if checkpoint else None
        )

 
