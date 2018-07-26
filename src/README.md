Source Scripts
===================
<details>
<summary><strong><em>Table of Contents</em></strong></summary>

<!-- TOC -->

- [Language Model Training with `Tensor2Tensor`](#language-model-training-with-tensor2tensor)
    - [Training on TPUs](#training-on-tpus)
        - [Monitoring Progress with Tensorboard](#monitoring-progress-with-tensorboard)
        - [Useful Resources about Using TPUs for Training with the `Tensor2Tensor` library](#useful-resources-about-using-tpus-for-training-with-the-tensor2tensor-library)
        - [Useful Tips about the Transformer Architecture](#useful-tips-about-the-transformer-architecture)

<!-- /TOC -->

</details>

There are various scripts here that are used for training the models and running the experiments. I'll try to keep this updated with latest instructions.

# Language Model Training with `Tensor2Tensor`

The first task we examine is language model training using [Transformer](https://arxiv.org/abs/1706.03762) models. Some useful resources on Transformers:

- [Training tips for transformer model](https://arxiv.org/pdf/1804.00247.pdf)
	* tips from using transformer for NMT Czesh <-> English
- [Harvard NLP review of transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
	* [repo](https://github.com/harvardnlp/annotated-transformer)
	* [annotated transformer on colab](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)
- [illustrated transformer](https://jalammar.github.io/illustrated-transformer/)

## Training on TPUs

I used the [`ctpu`](https://cloud.google.com/tpu/docs/quickstart) utility for launching and accessing TPUs. The script I used is saved in [gcloud-ctpu-startup.sh](scripts/gcloud-ctpu-startup.sh). 

You can use `ctpu` directly from the google cloud shell or install it from [source](https://github.com/tensorflow/tpu/tree/master/tools/ctpu), which works on Linux, macOS and WSL.

Set your appropriate region and start a VM.

```bash
gcloud config set compute/zone us-central1-f
./ctpu up
```

Since training can take a long time, I used `tmux` to start my session. You can find my [dotfiles online](https://github.com/akzaidi/etc/tree/master/dotfiles).

This will automatically SSH you into your host VM (this is not the TPU VM, which you cannot access directly).

Create a bucket and save some variables to access. You'll also need temporary storage on your host VM.

```bash
STORAGE_BUCKET=gs://alizaidi-tpu-data
DATA_DIR=$STORAGE_BUCKET/data/wikitext
TMP_DIR=/mnt/disks/mnt-dir/t2t_tmp/wikitext
mkdir /mnt/disks/mnt-dir/t2t_tmp
```

Examine your TPU's entry points.

```bash
gcloud compute tpus list --zone us-central1-f
```

Next we set up our connection strings to our TPU server:


```bash
gcloud compute tpus list --zone us-central1-f
TPU_IP=10.240.1.2
TPU_MASTER=grpc://$TPU_IP:8470
```

Use the data generator and language model specification for the [Wikitext-103 dataset](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/wikitext103.py):

```bash
t2t-datagen --problem=languagemodel_wikitext103 --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR
OUT_DIR=$STORAGE_BUCKET/training/transformer_lang_model/wikitext/
```

Train:

```bash
t2t-trainer \
  --model=transformer \
  --hparams_set=transformer_tpu \
  --problem=languagemodel_wikitext103 \
  --train_steps=100000 \
  --eval_steps=8 \
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR \
  --use_tpu=True \
  --master=$TPU_MASTER
```

### Monitoring Progress with Tensorboard

Open up port 6006:

```bash
gcloud compute firewall-rules create tensorboard-port --allow tcp:6006
```

Launch tensorboard:

```bash
tensorboard --logdir=$OUT_DIR
```

Navigate to your external IP address and port 6006. You can find your external IP address by printing `gcloud compute instances list`.

### Useful Resources about Using TPUs for Training with the `Tensor2Tensor` library

- [tensor2tensor/cloud_tpu.md](https://github.com/tensorflow/tensor2tensor/blob/master/docs/cloud_tpu.md)
- Eyal Gruss has summarized his experiences with the [rough guide to running transformer on TPU](https://github.com/eyaler/transformer_tpu) that is a very good read.
- [tf estimator on tpu](https://cloud.google.com/tpu/docs/using-estimator-api)
- [Chris Young: ImageNet is the new MNIST](https://supercomputersfordl2017.github.io/Presentations/ImageNetNewMNIST.pdf) 

### Useful Tips about the Transformer Architecture

1. [Popel et. al - Training tips for the transformer model](https://arxiv.org/pdf/1804.00247.pdf)
    - warm-up scheme: increase learning rate by $\sqrt{(\text{number of gpus})}$
        * for bigger models, try bigger warm-up steps (16k)
    - Always use max GPUs and experiments sequentially, instead of multiple parallel experiments with singleGPU
    - averaging 10+ checkpoints improves BLEU
    - adafactor can process 30% more tokens than Adam
        - `hparams.optimizer = "Adafactor"`
1. [Issue #444: Use as large batch as possible](https://github.com/tensorflow/tensor2tensor/issues/444)
1. [Issue #280: Why Noam, and warm ups for learning rates](https://github.com/tensorflow/tensor2tensor/issues/280)
