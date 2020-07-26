from typing import Tuple, Dict, Union, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split
from torch.optim import Optimizer, Adam
from torch.nn import CrossEntropyLoss
from pytorch_lightning.core.lightning import LightningModule

class Model(LightningModule):

    def __init__(self,
                 input_path: str = None):
        super(Model, self).__init__()
        # REQUIRED
        # define dataset and layers
        self.dataset = Dataset(input_path)
        self.layer = nn.Module()

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
        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=4)
        return train_dataloader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        # OPTIONAL
        val_dataloader = DataLoader(self.val_dataset,
                                    sampler=RandomSampler(self.val_dataset),
                                    batch_size=4)
        return val_dataloader

    def configure_optimizers(self) -> Optional[
        Union[
            Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]
        ]
    ]:
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optimizer = Adam(self.parameters(),
                         lr=2e-5)
        return optimizer

    def training_step(self, batch, batch_idx) -> Union[
        int, Dict[
            str, Union[
                Tensor, Dict[str, Tensor]
            ]
        ]
    ]:
        # REQUIRED
        logits = self.forward(batch)
        labels = batch['label']
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        logs = {'train_loss': loss}

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        # OPTIONAL
        logits = self.forward(batch)
        labels = batch['label']
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return {'val_loss': loss}

    def validation_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        # REQUIRED
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        logs = {'avg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': logs}

