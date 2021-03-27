from typing import Tuple, List, Dict, Union
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from hydra.experimental import initialize, compose
from omegaconf import DictConfig

import pytorch_lightning as pl


class LightningBase(pl.LightningModule):

    def __init__(
        self,
        task_name: str,
        dir_path: str,
        batch_size: int,
        lr: float,
        num_gpus: int,
        num_tpus: int,
        weight_decay: float,
        training_step: int,
        warmup_step: int,
        accumulate_grad_batches: Union[int, Dict[int, int]],
        gradient_clip_val: float,
        accelerator: str,
        precision: int,
        max_len: int,
        val_check_interval: float,
        monitor: str,
        use_early_stopping: bool,
        use_tensor_board: bool,
    ) -> None:
        """
        Constructor of Lightning Base Class
        Args:
            task_name (str): task name for tensorboard
            dir_path (str): directory path to save models
            batch_size (int): batch size
            lr (float): learning rate
            num_gpus (int): number of gpus
            weight_decay (float): weight decay (L2 regularization)
            training_step (int): number of training steps
            warmup_step (int): number of warmup steps
            accumulate_grad_batches (float): factor of gradient accumulation
            gradient_clip_val (float): factor of gradient clipping
            accelerator (str): distributed training accelerator (e.g. 'dp', 'ddp', 'ddp2', 'horovod')
            precision (int): precision when model training (e.g. 16 => FP16, 32 => FP32)
            val_check_interval (float): steps interval you want to check model
            monitor (str): criteria of monitoring for callbacks
            use_early_stopping (bool): whether use or not early stopping
            use_tensor_board (bool): whether use or not tensor board
        """

        super().__init__()
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.lr = lr
        self.num_gpus = num_gpus
        self.num_tpus = num_tpus
        self.weight_decay = weight_decay
        self.training_step = training_step
        self.warmup_step = warmup_step
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.accelerator = accelerator
        self.precision = precision
        self.max_len = max_len
        self.val_check_interval = val_check_interval
        self.monitor = monitor
        self.use_early_stopping = use_early_stopping
        self.use_tensor_board = use_tensor_board
        self.task_name = task_name

        if self.precision < 32:
            self.use_amp = True

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> None:
        """
        Start to train the model.

        Args:
            train_dataloader (DataLoader): training dataloader
            val_dataloader (DataLoader): validation dataloader
        """

        callbacks: list = [
            ModelCheckpoint(
                monitor=self.monitor,
                dirpath=self.dir_path,
                filename=f"model.{self.global_step}",
            )
        ]

        if self.use_early_stopping:
            callbacks.append(EarlyStopping(monitor=self.monitor))

        if self.use_tensor_board:
            logger = TensorBoardLogger(self.dir_path, name=self.task_name)
        else:
            logger = True

        trainer = Trainer(
            gpus=self.num_gpus,
            distributed_backend=self.accelerator,
            precision=self.precision,
            amp_backend="apex" if self.use_amp else None,
            val_check_interval=self.val_check_interval,
            callbacks=callbacks,
            logger=logger,
            accumulate_grad_batches=self.accumulate_grad_batches,
            gradient_clip_val=self.gradient_clip_val,
        )

        trainer.fit(
            model=self,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LambdaLR]]:
        """
        Configure optimizers and lr schedulers
        Returns:
            (Tuple[List[Optimizer], List[LambdaLR]]): [optimizers], [schedulers]
        """

        no_decay = ["bias", "LayerNorm.weight"]
        model = self.model
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            params=optimizer_grouped_parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            self.training_step,
            self.warmup_step,
        )

        return [self.optimizer], [self.scheduler]

    @staticmethod
    def load_args(cfg_path: str, cfg_name: str) -> DictConfig:
        """
        Load arguments from yaml files using hydra
        Args:
            cfg_path (str): parents path (e.g. '../configs')
            cfg_name (str): config file name (e.g. 'bart_for_paraphrase_generation')
        Returns:
            (DictConfig): hydra configuration object
        """

        initialize(config_path=cfg_path)
        cfg = compose(config_name=cfg_name)
        return cfg
