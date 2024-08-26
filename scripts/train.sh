for MODEL_NAME in gpt4 midm_10b_plm_70k midm_v3_normal_adjusted midm_v3_only_josa_adjusted
do
accelerate launch train.py --train_data ./data/klue_nli_train.jsonl --val_data ./data/klue_nli_dev.jsonl --logging_term 100 --output_dir ../kt_output/$MODEL_NAME/nli --epochs 5 --batch_size 64 --eval_batch_size 1 --lr 2e-5 --decay 0.1 --gradient_clipping 1. --warmup 0 --accumulation_steps 1 --plm_path /home/work/g-earth-22/jhlee/git/midm_proxy/hf_checkpoints/$MODEL_NAME/step_24500 --add_eos_token --task nli --generate --max_new_tokens 8 --use_flash_attention_2
done

MODEL_NAME=llama3.1
accelerate launch train.py --train_data ./data/klue_nli_train.jsonl --val_data ./data/klue_nli_dev.jsonl --logging_term 100 --output_dir ../kt_output/$MODEL_NAME/nli --epochs 5 --batch_size 64 --eval_batch_size 1 --lr 2e-5 --decay 0.1 --gradient_clipping 1. --warmup 0 --accumulation_steps 1 --plm_path /home/work/g-earth-22/jhlee/git/midm_proxy/hf_checkpoints/$MODEL_NAME/step_24500 --add_bos_token --add_eos_token --task nli --generate --max_new_tokens 8 --use_flash_attention_2
