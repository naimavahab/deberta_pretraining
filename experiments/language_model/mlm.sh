#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

cache_dir=/home/ec2-user/deberta_pretraining/experiments/language_model

max_seq_length=64
data_dir=$cache_dir/rna_data/spm_$max_seq_length #wiki103/spm_$max_seq_length

function setup_wiki_data(){
	task=$1
	mkdir -p $cache_dir
	#if [[ ! -e  $cache_dir/spm.model ]]; then
	#	wget -q https://huggingface.co/microsoft/deberta-v3-base/resolve/main/spm.model -O $cache_dir/spm.model
#	fi

	if [[ ! -e  $data_dir/test.txt ]]; then
	#	wget -q https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -O $cache_dir/wiki103.zip
	#	unzip -j $cache_dir/wiki103.zip -d $cache_dir/wiki103
		mkdir -p $data_dir
		python3 ./prepare_data.py -i $cache_dir/rna_data/lnc_train1.txt -o $data_dir/train.txt --max_seq_length $max_seq_length
#		python3 ./prepare_data.py -i $cache_dir/wikitext-103/wiki.valid.tokens -o $data_dir/valid.txt --max_seq_length $max_seq_length
#		python3 ./prepare_data.py -i $cache_dir/wikitext-103/wiki.test.tokens -o $data_dir/test.txt --max_seq_length $max_seq_length
		python3 ./prepare_data.py -i $cache_dir/rna_data/lnc_test1.txt -o $data_dir/test.txt --max_seq_length $max_seq_length
		python3 ./prepare_data.py -i $cache_dir/rna_data/lnc_valid1.txt -o $data_dir/valid.txt --max_seq_length $max_seq_length
	fi
}

setup_wiki_data

Task=MLM

init=$1
tag=$init
case ${init,,} in
	bert-base)
	parameters=" --num_train_epochs 1 \
	--model_config bert_base.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 256 \
	--max_ngram 1 \
	--fp16 True "
		;;
	deberta-base)
	parameters=" --num_train_epochs 1 \
	--model_config deberta_base.json \
	--warmup 100 \
	--learning_rate 1e-4 \
	--train_batch_size 8 \
	--max_ngram 3 \
	--eval_batch_size 8 \
	--predict_batch_size 8 \
	--fp16 True "
		;;
	xlarge-v2)
	parameters=" --num_train_epochs 1 \
	--model_config deberta_xlarge.json \
	--warmup 1000 \
	--learning_rate 1e-4 \
	--train_batch_size 32 \
	--max_ngram 3 \
	--fp16 True "
		;;
	xxlarge-v2)
	parameters=" --num_train_epochs 1 \
	--warmup 1000 \
	--model_config deberta_xxlarge.json \
	--learning_rate 1e-4 \
	--train_batch_size 32 \
	--max_ngram 3 \
	--fp16 True "
		;;
	*)
		echo "usage $0 <Pretrained model configuration>"
		echo "Supported configurations"
		echo "bert-base - Pretrained a bert base model with DeBERTa vocabulary (12 layers, 768 hidden size, 128k vocabulary size)"
		echo "deberta-base - Pretrained a deberta base model (12 layers, 768 hidden size, 128k vocabulary size)"
		echo "xlarge-v2 - Pretrained DeBERTa v2 model with 900M parameters (24 layers, 1536 hidden size)"
		echo "xxlarge-v2 - Pretrained DeBERTa v2 model with 1.5B parameters (48 layers, 1536 hidden size)"
		exit 0
		;;
esac

python3 -m DeBERTa.apps.run \
	--tag $tag \
	--do_train \
	--num_training_steps 1000 \
	--max_seq_len $max_seq_length \
	--dump 10000 \
	--task_name $Task \
	--data_dir $data_dir \
	--vocab_path $cached_dir/spm.model \
	--vocab_type spm \
	--output_dir /home/ec2-user/deberta_pretraining/experiments/language_model/$tag/$task  $parameters
