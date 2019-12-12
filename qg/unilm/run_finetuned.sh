DATA_DIR=/var/services/homes/miksingh/nlp/workspace/microsoft/code/unilm/src/test
MODEL_RECOVER_PATH=/var/services/homes/miksingh/nlp/workspace/microsoft/code/unilm/src/qg_model.bin
EVAL_SPLIT=test
# run decoding
python biunilm/decode_seq2seq.py --bert_model bert-large-cased --new_segment_ids --mode s2s \
  --input_file ${DATA_DIR}/test.pa.tok.txt --split ${EVAL_SPLIT} --tokenized_input \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 512 --max_tgt_length 48 \
  --batch_size 16 --beam_size 1 --length_penalty 0
# run evaluation using our tokenized data as reference
python2 qg/eval_on_unilm_tokenized_ref.py --out_file qg/output/qg.test.output.txt
# run evaluation using tokenized data of Du et al. (2017) as reference
python2 qg/eval.py --out_file qg/output/qg.test.output.txt
