#!/usr/bin/env bash

. ./path.sh || exit 1;

# Configuración de la máquina
CUDA_VISIBLE_DEVICES="0"
gpu_num=1
count=1
gpu_inference=true  # Para correr el decoding con gpu, cambiar a falso para cpu decoding
# Para gpu decoding, inference_nj=ngpu*njob; para cpu decoding, inference_nj=njob
njob=1
train_cmd=utils/run.pl
infer_cmd=utils/run.pl

# Configuración general
feats_dir="../DATA" #Feature output dictionary
exp_dir="."
lang=zh
#token_type=char
token_type=word
type=sound
scp=wav.scp
speed_perturb="0.9 1.0 1.1"
stage=0
stop_stage=5

# Feature configuration
feats_dim=30
nj=64
#nj=8

# data
raw_data=/home/jose/Documents/TP2/raw_data
# TO DO: DELETE
data_url=www.openslr.org/resources/33

# exp tag
tag="exp1"

. ../transformer/utils/parse_options.sh || exit 1;

# Configurar bash en 'debug' mode, se mostrará la salida :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="dev test"

asr_config=conf/train_asr_paraformer_conformer_12e_6d_2048_256.yaml
model_dir="baseline_$(basename "${asr_config}" .yaml)_${lang}_${token_type}_${tag}"

inference_config=conf/decode_asr_transformer_noctc_1best.yaml
inference_asr_model=valid.acc.ave_10best.pb
#inference_asr_model=valid.acc.ave_1best.pb

# Puedes configurar el número de gpus aquí
gpuid_list=$CUDA_VISIBLE_DEVICES  # Configurar gpus para decoding, igual que el estado de entrenamiento por defecto
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

if ${gpu_inference}; then
    inference_nj=$[${ngpu}*${njob}]
    _ngpu=1
else
    inference_nj=$njob
    _ngpu=0
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download_and_untar.sh ${raw_data} ${data_url} data_aishell
    local/download_and_untar.sh ${raw_data} ${data_url} resource_aishell
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
     echo "stage 0: Data preparation"
    # Data preparation
    local/aishell_data_prep.sh ${raw_data}/data_aishell/wav ${raw_data}/data_aishell/transcript ${feats_dir}
    for x in train dev test; do
        cp ${feats_dir}/data/${x}/text ${feats_dir}/data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " ${feats_dir}/data/${x}/text.org) <(cut -f 2- -d" " ${feats_dir}/data/${x}/text.org | tr -d " ") \
            > ${feats_dir}/data/${x}/text
        #../transformer/utils/text2token.py -n 1 -s 1 ${feats_dir}/data/${x}/text > ${feats_dir}/data/${x}/text.org
        mv ${feats_dir}/data/${x}/text.org ${feats_dir}/data/${x}/text
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
     echo "stage 1: Feature and CMVN Generation"
    ../transformer/utils/compute_cmvn.sh --fbankdir ${feats_dir}/data/${train_set} --cmd "$train_cmd" --nj $nj --feats_dim ${feats_dim} --config_file "$asr_config" --scale 1.0
fi

token_list=${feats_dir}/data/${lang}_token_list/$token_type/tokens.txt
echo "dictionary: ${token_list}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
     echo "stage 2: Dictionary Preparation"
    mkdir -p ${feats_dir}/data/${lang}_token_list/$token_type/
   
    echo "make a dictionary"
    echo "<blank>" > ${token_list}
    echo "<s>" >> ${token_list}
    echo "</s>" >> ${token_list}
    ../transformer/utils/text2token.py -s 1 -n 1 --space "" ${feats_dir}/data/$train_set/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0}' >> ${token_list}
    echo "<unk>" >> ${token_list}
fi