export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
device=$1
lr=$2
epochs=$3
wp=$4
name=english_t5_epochs${epochs}_lr${lr}_d4_wp${wp}_wd0_acc1_bs32
CUDA_VISIBLE_DEVICES=$device nohup /share/miniconda3/envs/llama/bin/python run_s2s.py \
--train_path ours_dataset_v4/en/train.jsonl \
--hc3_val_path ours_dataset_v4/en/val_hc3.jsonl \
--ours_val_path ours_dataset_v4/en/val_ours.jsonl \
--model allenai/tk-instruct-base-def-pos \
--max_length 512 \
--batch_size 32 \
--save_path model/$name \
--tensorboard_dir tflog/$name \
--num_test_times 10 \
--lang en \
--epochs $epochs \
--lr $lr \
--seed 42 \
--warm_up_ratio $wp \
--weight_decay 0.0 \
--accumulation_steps 1 \
>log/${name}.log 2>&1 &