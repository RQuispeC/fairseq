total_updates=20000
warmup_updates=500
lr=0.001
max_tokens=256
update_freq=4
pointer_layer=-2

CUDA_VISIBLE_DEVICES=0 \
    fairseq-train examples/pointer_generator/bin \
    --max-tokens "$max_tokens" \
    --task translation \
    --source-lang src --target-lang tgt \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --required-batch-size-multiple 1 \
    --arch transformer_pointer_generator \
    --alignment-layer "$pointer_layer" \
    --alignment-heads 1 \
    --source-position-markers 1000 \
    --criterion cross_entropy \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler inverse_sqrt --lr "$lr" --max-update "$total_updates" --warmup-updates "$warmup_updates" \
    --update-freq "$update_freq" \
    --skip-invalid-size-inputs-valid-test \
    --no-token-positional-embeddings
#     --label-smoothing 0.1 \
#    --user-dir examples/pointer_generator/pointer_generator_src \
    
