from absl import app, flags, logging

import torch
from pytorch_lightning import Trainer

from model import Model


FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', default=None,
                    help='Path to the training dataset')
flags.DEFINE_integer('cuda_device', default=0,
                     help='If given, uses this CUDA device in training')


def main(argv):
    model = Model(input_path=FLAGS.input_path)

    # most basic trainer, uses good defaults (1 gpu)
    trainer = Trainer(gpus=FLAGS.cuda_device)
    logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
    logging.info(f'Use the number of GPU: {FLAGS.cuda_device}')
    trainer.fit(model)


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'input_path'
    ])
    app.run(main)