Source Scripts
===================

There are various scripts here that are used for training the models and running the experiments. I'll try to keep this updated with latest instructions.

## Language Model Training with `Tensor2Tensor`

The first task we examine is language model training using [Transformer](https://arxiv.org/abs/1706.03762) models. Some useful resources on Transformers:


### Training on TPUs


#### Useful Resources about Using TPUs for Training with the `Tensor2Tensor` library

- [tensor2tensor/cloud_tpu.md](https://github.com/tensorflow/tensor2tensor/blob/master/docs/cloud_tpu.md)
- Eyal Gruss has summarized his experiences with the [rough guide to running transformer on TPU](https://github.com/eyaler/transformer_tpu) that is a very good read.
- [tf estimator on tpu](https://cloud.google.com/tpu/docs/using-estimator-api)
- [Chris Young: ImageNet is the new MNIST](https://supercomputersfordl2017.github.io/Presentations/ImageNetNewMNIST.pdf) 

#### Useful Tips about the Transformer Architecture

1. [Popel et. al - Training tips for the transformer model](https://arxiv.org/pdf/1804.00247.pdf)
    - warm-up scheme: increase learning rate by $\sqrt{(\text{number of gpus})}$
        * for bigger models, try bigger warm-up steps (16k)
    - Always use max GPUs and experiments sequentially, instead of multiple parallel experiments with singleGPU
    - averaging 10+ checkpoints improves BLEU
    - adafactor can process 30% more tokens than Adam
        - `hparams.optimizer = "Adafactor"`
1. [Issue #444: Use as large batch as possible](https://github.com/tensorflow/tensor2tensor/issues/444)
1. [Issue #280: Why Noam, and warm ups for learning rates](https://github.com/tensorflow/tensor2tensor/issues/280)
