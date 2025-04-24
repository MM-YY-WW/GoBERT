for mask in {1,2,3,4,5}
do
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_port=12372 gobert.py \
    --train_file "data/unique_train_311536.txt" \
    --validation_file "data/unique_test_47160.txt" \
    --use_slow_tokenizer \
    --tokenizer_name "/home/yuwei/miao/GoBERT/go_bert/extrago/full3_tune/tokenizer_epoch_17" \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --learning_rate 2.1e-4 \
    --weight_decay 0.01 \
    --num_train_epochs 18 \
    --gradient_accumulation_steps 1 \
    --custom_scheduler onecycle \
    --num_warmup_steps 70 \
    --output_dir "ckpt/evaluation_result" \
    --seed 0 \
    --model_type bert \
    --max_seq_length 50 \
    --line_by_line True \
    --semantic_path "" \
    --mlm_probability 0.15 \
    --checkpointing_steps 100 \
    --preprocessing_num_workers 15 \
    --mask 1 \
    --mask_seed $mask \
    --eval_only \
    --resume_from_checkpoint /home/yuwei/miao/GoBERT/go_bert/extrago/full3_tune/epoch_17
done