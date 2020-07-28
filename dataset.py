import logging
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class Dataset(Dataset):
    def __init__(self,
                 input_path: str = None)

        logger.info(f'Reading file at {file_path}')

        with open(input_path) as dataset_file:
            self.dataset = dataset_file.readlines()

        logger.info('Reading the dataset')

        self.processed_dataset = []

        for line in tqdm(self.dataset, desc='Processing'):
            data = json.loads(line)
            processed_data = {}

            processed_data['feature'] = torch.Tensor([data['feature']])

            processed_data['label'] = torch.LongTensor([data['label']])

            self.processed_dataset.append((processed_data))

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self,
                    idx: int = None):
        return self.processed_dataset[idx]