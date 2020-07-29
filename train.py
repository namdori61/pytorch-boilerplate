from absl import app, flags, logging

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from model import Model


FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', default=None,
                    help='Path to the training dataset')
flags.DEFINE_string('save_dir', default=None,
                    help='Path to save model')
flags.DEFINE_integer('cuda_device', default=0,
                     help='If given, uses this CUDA device in training')
flags.DEFINE_integer('batch_size', default=4,
                     help='If given, uses this batch size in training')
flags.DEFINE_integer('num_workers', default=0,
                     help='If given, uses this number of workers in data loading')
flags.DEFINE_float('lr', default=2e-5,
                   help='If given, uses this learning rate in training')
flags.DEFINE_integer('seed', default=0,
                     help='If given, uses this number as random seed in training and become deterministic')


def main(argv):
    model = Model(input_path=FLAGS.input_path,
                  batch_size=FLAGS.batch_size,
                  num_workers=FLAGS.num_workers,
                  lr=FLAGS.lr)
    if FLAGS.seed != 0:
        seed_everything(FLAGS.seed)

    checkpoint_callback = ModelCheckpoint(
        filepath=FLAGS.save_dir,
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='model.ckpt'
    )

    # most basic trainer, uses good defaults (1 gpu)
    if FLAGS.cuda_device > 1:
        trainer = Trainer(deterministic=True,
                          gpus=FLAGS.cuda_device,
                          distributed_backend='ddp',
                          log_gpu_memory=True,
                          checkpoint_callback=checkpoint_callback)
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info(f'Use the number of GPU: {FLAGS.cuda_device}')
    elif FLAGS.cuda_device == 1:
        trainer = Trainer(deterministic=True,
                          gpus=FLAGS.cuda_device,
                          log_gpu_memory=True,
                          checkpoint_callback=checkpoint_callback)
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info(f'Use the number of GPU: {FLAGS.cuda_device}')
    else:
        trainer = Trainer(deterministic=True,
                          checkpoint_callback=checkpoint_callback)
        logging.info('No GPU available, using the CPU instead.')
    trainer.fit(model)

if __name__ == '__main__':
    flags.mark_flags_as_required([
        'input_path'
    ])
    app.run(main)