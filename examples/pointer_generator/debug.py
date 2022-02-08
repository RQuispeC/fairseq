# %%
import argparse
from argparse import Namespace
import logging
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils
from fairseq.data import data_utils, iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer

# %%
args = Namespace(activation_dropout=0.0, activation_fn='relu', adam_betas='(0.9, 0.999)', adam_eps=1e-08, adaptive_input=False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, alignment_heads=1, alignment_layer=4, all_gather_list_size=16384, amp=False, amp_batch_retries=2, amp_init_scale=128, amp_scale_window=None, arch='transformer_pointer_generator', attention_dropout=0.1, azureml_logging=False, batch_size=None, batch_size_valid=None, best_checkpoint_metric='loss', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_activations=False, checkpoint_shard_count=1, checkpoint_suffix='', clip_norm=0.1, combine_valid_subsets=None, cpu=False, cpu_offload=False, criterion='label_smoothed_cross_entropy', cross_self_attention=False, curriculum=0, data='./bin', data_buffer_size=10, dataset_impl=None, ddp_backend='pytorch_ddp', ddp_comm_hook='none', decoder_attention_heads=8, decoder_embed_dim=512, decoder_embed_path=None, decoder_ffn_embed_dim=2048, decoder_input_dim=512, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=False, decoder_normalize_before=True, decoder_output_dim=512, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_num_procs=1, distributed_port=-1, distributed_rank=0, distributed_world_size=1, dropout=0.1, ema_decay=0.9999, ema_fp32=False, ema_seed_model=None, ema_start_update=0, ema_update_freq=1, empty_cache_freq=0, encoder_attention_heads=8, encoder_embed_dim=512, encoder_embed_path=None, encoder_ffn_embed_dim=2048, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=False, encoder_normalize_before=True, eos=2, eval_bleu=False, eval_bleu_args='{}', eval_bleu_detok='space', eval_bleu_detok_args='{}', eval_bleu_print_samples=False, eval_bleu_remove_bpe=None, eval_tokenized_bleu=False, fast_stat_sync=False, find_unused_parameters=False, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, force_generation=None, fp16=False, fp16_adam_stats=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, fp32_reduce_scatter=False, gen_subset='test', gradient_as_bucket_view=False, grouped_shuffling=False, heartbeat_timeout=-1, ignore_prefix_size=0, ignore_unused_valid_subsets=False, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_interval_updates_pattern=-1, keep_last_epochs=-1, label_smoothing=0.1, layernorm_embedding=True, left_pad_source=True, left_pad_target=False, load_alignments=False, load_checkpoint_on_all_dp_ranks=False, localsgd_frequency=3, log_file=None, log_format=None, log_interval=100, lr=[0.001], lr_scheduler='inverse_sqrt', max_epoch=0, max_tokens=256, max_tokens_valid=256, max_update=20000, max_valid_steps=None, maximize_best_checkpoint_metric=False, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=False, no_reshard_after_forward=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_seed_provided=False, no_token_positional_embeddings=False, not_fsdp_flatten_parameters=False, nprocs_per_node=1, num_batch_buckets=0, num_shards=1, num_workers=1, offload_activations=False, on_cpu_convert_precision=False, optimizer='adam', optimizer_overrides='{}', pad=1, patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, plasma_path='/tmp/plasma', profile=False, quant_noise_pq=0, quant_noise_pq_block_size=8, quant_noise_scalar=0, quantization_config_path=None, report_accuracy=False, required_batch_size_multiple=1, required_seq_len_multiple=1, reset_dataloader=False, reset_logging=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, restore_file='checkpoint_last.pt', save_dir='checkpoints', save_interval=1, save_interval_updates=0, scoring='bleu', seed=1, sentence_avg=False, shard_id=0, share_all_embeddings=True, share_decoder_input_output_embed=False, skip_invalid_size_inputs_valid_test=True, skip_remainder_batch=False, slowmo_base_algorithm='localsgd', slowmo_momentum=None, source_lang='src', source_position_markers=1000, stop_min_lr=-1.0, stop_time_hours=0, store_ema=False, suppress_crashes=False, target_lang='tgt', task='translation', tensorboard_logdir=None, threshold_loss_scale=None, tie_adaptive_weights=False, tokenizer=None, tpu=False, train_subset='train', truncate_source=True, unk=3, update_epoch_batch_itr=False, update_freq=[4], update_ordered_indices_seed=False, upsample_primary=-1, use_bmuf=False, use_old_adam=False, use_plasma_view=False, use_sharded_state=False, user_dir=None, valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, wandb_project=None, warmup_init_lr=-1, warmup_updates=500, weight_decay=0.01, write_checkpoints_asynchronously=False, zero_sharding='none')
cfg = convert_namespace_to_omegaconf(args)

# %%
task = tasks.setup_task(cfg.task)
# %%
model = task.build_model(cfg.model)
# %%
# Seems all model parameters are trainable
for i in model.named_parameters():
    if not i[1].requires_grad:
        print(i[0])
# %%
print("num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

print("num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )
# %%
criterion = task.build_criterion(cfg.criterion)
# %%
