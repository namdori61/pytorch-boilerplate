from typing import Tuple, Dict, Union, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, random_split
from torch.optim import Optimizer, Adam
from torch.nn import CrossEntropyLoss
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy, precision, recall

from dataset import CumstomDataset

class Model(LightningModule):

    def __init__(self,
                 input_path: str = None,
                 num_classes: int = 2,
                 batch_size: int = 4,
                 num_workers: int = 0,
                 lr: float = 2e-5,
                 cuda_device: int = 0):
        super(Model, self).__init__()
        # REQUIRED
        # define dataset and layers
        self.dataset = CumstomDataset(input_path)
        self.layer = nn.Module()

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.cuda_device = cuda_device # Number of gpu to use, not gpu index

    def forward(self,
                batch: Union[Tensor, Dict] = None) -> Union[float, Dict]:
        # REQUIRED
        # forward propagation
        output = self.layer(batch)
        return output

    def setup(self,
              step):
        # OPTIONAL
        # split dataset into train and validation
        train_size = int(len(self.dataset) * 0.8)
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        # REQUIRED
        if self.cuda_device > 0:
            sampler = DistributedSampler(self.train_dataset)
        else:
            sampler = RandomSampler(self.train_dataset)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=sampler,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers)
        return train_dataloader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        # OPTIONAL
        if self.cuda_device > 0:
            sampler = DistributedSampler(self.val_dataset)
        else:
            sampler = RandomSampler(self.val_dataset)

        val_dataloader = DataLoader(self.val_dataset,
                                    sampler=sampler,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers)
        return val_dataloader

    def configure_optimizers(self) -> Optional[
        Union[
            Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]
        ]
    ]:
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optimizer = Adam(self.parameters(),
                         lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx) -> Union[
        int, Dict[
            str, Union[
                Tensor, Dict[str, Tensor]
            ]
        ]
    ]:
        logits = self.forward(batch)
        labels = batch['label']
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels.view(-1), num_classes=self.num_classes)

        train_logs = {'train_loss': loss, 'train_accuracy': acc}

        return {'loss': loss, 'log': train_logs}

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        logits = self.forward(batch)
        labels = batch['label']
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        preds = torch.argmax(logits, dim=1)
        val_acc = accuracy(preds, labels.view(-1), num_classes=self.num_classes)
        val_pr = precision(preds, labels.view(-1), num_classes=self.num_classes)
        val_rc = recall(preds, labels.view(-1), num_classes=self.num_classes)

        return {'val_loss': loss,
                'val_acc': val_acc,
                'val_pr': val_pr,
                'val_rc': val_rc}

    def validation_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_pr = torch.stack([x['val_pr'] for x in outputs]).mean()
        avg_rc = torch.stack([x['val_rc'] for x in outputs]).mean()

        logs = {'avg_val_loss': avg_loss,
                'avg_val_acc': avg_acc,
                'avg_val_pr': avg_pr,
                'avg_val_rc': avg_rc}
        return {'val_loss': avg_loss, 'log': logs}

