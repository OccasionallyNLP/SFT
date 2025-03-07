# QA midm_10b_plm_70k midm_v3_normal_adjusted midm_v3_only_josa_adjusted
for MODEL_NAME in gpt4 
do
accelerate launch inference.py --test_data /home/work/g-earth-22/hoyoun/downstream/data/KorQuad_dev.jsonl --output_dir ../kt_output/$MODEL_NAME/qa/model_2 --batch_size 1 --model_path ../kt_output/$MODEL_NAME/qa/model_2 --use_flash_attention_2 --max_new_tokens=64
done

# MODEL_NAME=llama3.1
# TASK=qa
# accelerate launch train.py --train_data /home/work/g-earth-22/hoyoun/downstream/data/KorQuad_train.jsonl --val_data /home/work/g-earth-22/hoyoun/downstream/data/KorQuad_dev.jsonl --logging_term 100 --output_dir ../kt_output/$MODEL_NAME/$TASK --epochs 2 --batch_size 1 --eval_batch_size 1 --lr 2e-5 --decay 0.1 --gradient_clipping 1. --warmup 0 --accumulation_steps 8 --plm_path /home/work/g-earth-22/jhlee/git/midm_proxy/hf_checkpoints/$MODEL_NAME/step_24500 --add_eos_token --task $TASK --use_flash_attention_2 --save_model_every_epoch
