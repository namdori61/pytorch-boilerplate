# pytorch-boilerplate
Pytorch boilerplate with Pytorch-Lightning, Optuna, and so on

## Pytorch-Lightning

1. How to train

- `python YOUR_PATH/train.py --input_path YOUR_PATH/DATA --version VERSION_NAME --lr LR  --cuda_device NO_CUDA --max_epochs EPOCHS --save_dir YOUR_PATH/DIR --batch_size BATCH_SIZE --num_workers NO_WORKERS --seed SEED_TO_USE`
- Example: `python project/train.py --input_path project/data/data.txt --version lr5e-5_ep5_bc8 --lr 5e-5  --cuda_device 4 --max_epochs 5 --save_dir project/logs --batch_size 8 --num_workers 5`

2. Usage of Tensorboard logger
- Activate logger dashboard : `tensorboard --host 0.0.0.0 --logdir`
- Connect to dashboard with web-browser : host_ip:6006

## Optuna

## Alternatives

### Pytorch Wrapper
1. Pytorch-Lightning
2. Pytorch-Ignite
3. FAST-AI

### Hyper-parameter Optimization
1. Optuna
2. NNI
3. Hydra
4. Neptune
5. Ray Tune

### Configuration Management
1. omega conf

## References
1. Docs : https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html#
2. Github tutorial MNIST: https://github.com/PyTorchLightning/pytorch-lightning-conference-seed/tree/master/src/research_mnist 
3. Lightning-demo MNIST : https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=HOk9c4_35FKg