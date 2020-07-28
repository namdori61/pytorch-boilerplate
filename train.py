from absl import app, flags, logging

import torch
from pytorch_lightning import Trainer

from model import Model


FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', default=None,
                    help='Path to the training dataset')
flags.DEFINE_integer('cuda_device', default=0,
                     help='If given, uses this CUDA device in training')
flags.DEFINE_integer('batch_size', default=4,
                     help='If given, uses this batch size in training')
flags.DEFINE_integer('num_workers', default=0,
                     help='If given, uses this number of workers in data loading')
flags.DEFINE_float('lr', default=2e-5,
                   help='If given, uses this learning rate in training')


def main(argv):
    model = Model(input_path=FLAGS.input_path,
                  batch_size=FLAGS.batch_size,
                  num_workers=FLAGS.num_workers,
                  lr=FLAGS.lr)

    # most basic trainer, uses good defaults (1 gpu)
    if FLAGS.cuda_device > 1:
        trainer = Trainer(gpus=FLAGS.cuda_device,
                          distributed_backend='ddp',
                          log_gpu_memory=True)
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info(f'Use the number of GPU: {FLAGS.cuda_device}')
    elif FLAGS.cuda_device == 1:
        trainer = Trainer(gpus=FLAGS.cuda_device,
                          log_gpu_memory=True)
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info(f'Use the number of GPU: {FLAGS.cuda_device}')
    else:
        trainer = Trainer()
        logging.info('No GPU available, using the CPU instead.')
    trainer.fit(model)

if __name__ == '__main__':
    flags.mark_flags_as_required([
        'input_path'
    ])
    app.run(main)