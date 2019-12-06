DATA_DIR=/var/services/homes/miksingh/nlp/workspace/microsoft/code/unilm/src/test
MODEL_RECOVER_PATH=/var/services/homes/miksingh/nlp/workspace/microsoft/code/unilm/src/qg_model.bin
EVAL_SPLIT=test
# run decoding
python biunilm/decode_seq2seq.py --bert_model bert-large-cased --new_segment_ids --mode s2s \
  --input_file ${DATA_DIR}/custom.pa.txt --split ${EVAL_SPLIT} \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 512 --max_tgt_length 48 \
  --batch_size 16 --beam_size 5 --length_penalty 0
