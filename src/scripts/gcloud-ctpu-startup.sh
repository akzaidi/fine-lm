cd ~/Google\ Drive/Conference/jeju/gcloud

gcloud config set compute/zone us-central1-f
./ctpu up

STORAGE_BUCKET=gs://alizaidi-tpu-data
DATA_DIR=$STORAGE_BUCKET/data/wikitext103
TMP_DIR=/mnt/disks/mnt-dir/t2t_tmp/wikitext103

mkdir /mnt/disks/mnt-dir/t2t_tmp/wikitext103

gcloud compute tpus list --zone us-central1-f
TPU_IP=10.240.1.2
TPU_MASTER=grpc://$TPU_IP:8470

t2t-datagen --problem=languagemodel_lm1b8k_packed --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR
t2t-datagen --problem=languagemodel_wikitext103 --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR
OUT_DIR=$STORAGE_BUCKET/training/transformer_lang_model/wikitext103


t2t-trainer \
  --model=transformer \
  --hparams_set=transformer_tpu \
  --problem=languagemodel_wikitext103 \
  --train_steps=100000 \
  --eval_steps=10 \
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR \
  --use_tpu=True \
  --master=$TPU_MASTER


t2t-trainer \
  --problem=languagemodel_lm1b8k_packed \
  --model=transformer \
  --hparams_set=transformer_tpu \
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR \
  --train_steps=100000 \
  --use_tpu=True \
  --cloud_mlengine_master_type=cloud_tpu \
  --cloud_mlengine \
  --hparams_range=transformer_base_range \
  --autotune_objective='metrics-languagemodel_lm1b8k_packed/neg_log_perplexity' \
  --autotune_maximize \
  --autotune_max_trials=100 \
  --autotune_parallel_trials=5


tensorboard --logdir=$OUT_DIR
tensorboard --logdir=gs://alizaidi-tpu-data/training/transformer_lang_model  

export MODEL_DIR=gs://alizaidi-tpu-data/training/transformer_lang_model
tensorboard --logdir=$MODEL_DIR --port=8080

gcloud compute firewall-rules create tensorboard --allow tcp:6006 --source-tags=alikaz --description="tensorboard-viewer"


`googleapiclient.errors.HttpError: <HttpError 400 when requesting https://ml.googleapis.com/v1/projects/******/jobs?alt=json returned "Field: master_type Error: The specified machine type for masteris not supported in TPU training jobs: cloud_tpu"`