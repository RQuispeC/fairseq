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

from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils, optim
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

from itertools import chain

# %%
# following uses label_smoothed_cross_entropy
args = Namespace(activation_dropout=0.0, activation_fn='relu', adam_betas='(0.9, 0.999)', adam_eps=1e-08, adaptive_input=False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, alignment_heads=1, alignment_layer=4, all_gather_list_size=16384, amp=False, amp_batch_retries=2, amp_init_scale=128, amp_scale_window=None, arch='transformer_pointer_generator', attention_dropout=0.1, azureml_logging=False, batch_size=None, batch_size_valid=None, best_checkpoint_metric='loss', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_activations=False, checkpoint_shard_count=1, checkpoint_suffix='', clip_norm=0.1, combine_valid_subsets=None, cpu=False, cpu_offload=False, criterion='label_smoothed_cross_entropy', cross_self_attention=False, curriculum=0, data='./bin', data_buffer_size=10, dataset_impl=None, ddp_backend='pytorch_ddp', ddp_comm_hook='none', decoder_attention_heads=8, decoder_embed_dim=512, decoder_embed_path=None, decoder_ffn_embed_dim=2048, decoder_input_dim=512, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=False, decoder_normalize_before=True, decoder_output_dim=512, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_num_procs=1, distributed_port=-1, distributed_rank=0, distributed_world_size=1, dropout=0.1, ema_decay=0.9999, ema_fp32=False, ema_seed_model=None, ema_start_update=0, ema_update_freq=1, empty_cache_freq=0, encoder_attention_heads=8, encoder_embed_dim=512, encoder_embed_path=None, encoder_ffn_embed_dim=2048, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=False, encoder_normalize_before=True, eos=2, eval_bleu=False, eval_bleu_args='{}', eval_bleu_detok='space', eval_bleu_detok_args='{}', eval_bleu_print_samples=False, eval_bleu_remove_bpe=None, eval_tokenized_bleu=False, fast_stat_sync=False, find_unused_parameters=False, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, force_generation=None, fp16=False, fp16_adam_stats=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, fp32_reduce_scatter=False, gen_subset='test', gradient_as_bucket_view=False, grouped_shuffling=False, heartbeat_timeout=-1, ignore_prefix_size=0, ignore_unused_valid_subsets=False, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_interval_updates_pattern=-1, keep_last_epochs=-1, label_smoothing=0.1, layernorm_embedding=True, left_pad_source=True, left_pad_target=False, load_alignments=False, load_checkpoint_on_all_dp_ranks=False, localsgd_frequency=3, log_file=None, log_format=None, log_interval=100, lr=[0.001], lr_scheduler='inverse_sqrt', max_epoch=0, max_tokens=128, max_tokens_valid=256, max_update=20000, max_valid_steps=None, maximize_best_checkpoint_metric=False, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=False, no_reshard_after_forward=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_seed_provided=False, no_token_positional_embeddings=False, not_fsdp_flatten_parameters=False, nprocs_per_node=1, num_batch_buckets=0, num_shards=1, num_workers=1, offload_activations=False, on_cpu_convert_precision=False, optimizer='adam', optimizer_overrides='{}', pad=1, patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, plasma_path='/tmp/plasma', profile=False, quant_noise_pq=0, quant_noise_pq_block_size=8, quant_noise_scalar=0, quantization_config_path=None, report_accuracy=False, required_batch_size_multiple=1, required_seq_len_multiple=1, reset_dataloader=False, reset_logging=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, restore_file='checkpoint_last.pt', save_dir='checkpoints', save_interval=1, save_interval_updates=0, scoring='bleu', seed=1, sentence_avg=False, shard_id=0, share_all_embeddings=True, share_decoder_input_output_embed=False, skip_invalid_size_inputs_valid_test=True, skip_remainder_batch=False, slowmo_base_algorithm='localsgd', slowmo_momentum=None, source_lang='src', source_position_markers=1000, stop_min_lr=-1.0, stop_time_hours=0, store_ema=False, suppress_crashes=False, target_lang='tgt', task='translation', tensorboard_logdir=None, threshold_loss_scale=None, tie_adaptive_weights=False, tokenizer=None, tpu=False, train_subset='train', truncate_source=True, unk=3, update_epoch_batch_itr=False, update_freq=[4], update_ordered_indices_seed=False, upsample_primary=-1, use_bmuf=False, use_old_adam=False, use_plasma_view=False, use_sharded_state=False, user_dir=None, valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, wandb_project=None, warmup_init_lr=-1, warmup_updates=500, weight_decay=0.01, write_checkpoints_asynchronously=False, zero_sharding='none')
# following uses cross_entropy
#args = Namespace(activation_dropout=0.0, activation_fn='relu', adam_betas='(0.9, 0.999)', adam_eps=1e-08, adaptive_input=False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, alignment_heads=1, alignment_layer=4, all_gather_list_size=16384, amp=False, amp_batch_retries=2, amp_init_scale=128, amp_scale_window=None, arch='transformer_pointer_generator', attention_dropout=0.1, azureml_logging=False, batch_size=None, batch_size_valid=None, best_checkpoint_metric='loss', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_activations=False, checkpoint_shard_count=1, checkpoint_suffix='', clip_norm=0.1, combine_valid_subsets=None, cpu=False, cpu_offload=False, criterion='cross_entropy', cross_self_attention=False, curriculum=0, data='./bin', data_buffer_size=10, dataset_impl=None, ddp_backend='pytorch_ddp', ddp_comm_hook='none', decoder_attention_heads=8, decoder_embed_dim=512, decoder_embed_path=None, decoder_ffn_embed_dim=2048, decoder_input_dim=512, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=False, decoder_normalize_before=True, decoder_output_dim=512, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_num_procs=1, distributed_port=-1, distributed_rank=0, distributed_world_size=1, dropout=0.1, ema_decay=0.9999, ema_fp32=False, ema_seed_model=None, ema_start_update=0, ema_update_freq=1, empty_cache_freq=0, encoder_attention_heads=8, encoder_embed_dim=512, encoder_embed_path=None, encoder_ffn_embed_dim=2048, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=False, encoder_normalize_before=True, eos=2, eval_bleu=False, eval_bleu_args='{}', eval_bleu_detok='space', eval_bleu_detok_args='{}', eval_bleu_print_samples=False, eval_bleu_remove_bpe=None, eval_tokenized_bleu=False, fast_stat_sync=False, find_unused_parameters=False, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, force_generation=None, fp16=False, fp16_adam_stats=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, fp32_reduce_scatter=False, gen_subset='test', gradient_as_bucket_view=False, grouped_shuffling=False, heartbeat_timeout=-1, ignore_prefix_size=0, ignore_unused_valid_subsets=False, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_interval_updates_pattern=-1, keep_last_epochs=-1, label_smoothing=0.1, layernorm_embedding=True, left_pad_source=True, left_pad_target=False, load_alignments=False, load_checkpoint_on_all_dp_ranks=False, localsgd_frequency=3, log_file=None, log_format=None, log_interval=100, lr=[0.001], lr_scheduler='inverse_sqrt', max_epoch=0, max_tokens=128, max_tokens_valid=256, max_update=20000, max_valid_steps=None, maximize_best_checkpoint_metric=False, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=False, no_reshard_after_forward=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_seed_provided=False, no_token_positional_embeddings=False, not_fsdp_flatten_parameters=False, nprocs_per_node=1, num_batch_buckets=0, num_shards=1, num_workers=1, offload_activations=False, on_cpu_convert_precision=False, optimizer='adam', optimizer_overrides='{}', pad=1, patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, plasma_path='/tmp/plasma', profile=False, quant_noise_pq=0, quant_noise_pq_block_size=8, quant_noise_scalar=0, quantization_config_path=None, report_accuracy=False, required_batch_size_multiple=1, required_seq_len_multiple=1, reset_dataloader=False, reset_logging=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, restore_file='checkpoint_last.pt', save_dir='checkpoints', save_interval=1, save_interval_updates=0, scoring='bleu', seed=1, sentence_avg=False, shard_id=0, share_all_embeddings=True, share_decoder_input_output_embed=False, skip_invalid_size_inputs_valid_test=True, skip_remainder_batch=False, slowmo_base_algorithm='localsgd', slowmo_momentum=None, source_lang='src', source_position_markers=1000, stop_min_lr=-1.0, stop_time_hours=0, store_ema=False, suppress_crashes=False, target_lang='tgt', task='translation', tensorboard_logdir=None, threshold_loss_scale=None, tie_adaptive_weights=False, tokenizer=None, tpu=False, train_subset='train', truncate_source=True, unk=3, update_epoch_batch_itr=False, update_freq=[4], update_ordered_indices_seed=False, upsample_primary=-1, use_bmuf=False, use_old_adam=False, use_plasma_view=False, use_sharded_state=False, user_dir=None, valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, wandb_project=None, warmup_init_lr=-1, warmup_updates=500, weight_decay=0.01, write_checkpoints_asynchronously=False, zero_sharding='none')

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
#model = model.cuda()
model.train()

# %%
params = list(
            filter(
                lambda p: p.requires_grad,
                chain(model.parameters(), criterion.parameters()),
            )
        )
optimizer = optim.build_optimizer(cfg.optimizer, params)
# %%
# import opacus
# privacy_engine = opacus.PrivacyEngine(module=model,
#         batch_size=16, sample_size=1000,
#         max_grad_norm=1.0, noise_multiplier=1.0, target_delta=1e-5
#     )
# %%
# privacy_engine.attach(optimizer)
# %%
# Load valid dataset (we load training data below, based on the latest checkpoint)
# We load the valid dataset AFTER building the model
data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
if cfg.dataset.combine_valid_subsets:
    task.load_dataset("valid", combine=True, epoch=1)
else:
    for valid_sub_split in cfg.dataset.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)
# %%
trainer = Trainer(cfg, task, model, criterion, None)

# %%
# Load the latest checkpoint if one is available and restore the
# corresponding train iterator
extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
    cfg.checkpoint,
    trainer,
    # don't cache epoch iterators for sharded datasets
    disable_iterator_cache=task.has_sharded_data("train"),
)

# %%
def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config

# %%
@metrics.aggregate("train")
def train(cfg, trainer, task, epoch_itr):
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(
        itr,
        update_freq,
        skip_remainder_batch=cfg.optimization.skip_remainder_batch,
    )
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            print(samples)
            #log_output = trainer.train_step(samples)
            return samples


# %%
samples = train(cfg, trainer, task, epoch_itr)
# %%
trainer.train_step(samples)
# %%
# fairseq_task.py has this code that is called by trainer.train_step
sample, is_dummy_batch = trainer._prepare_sample(samples[0])
with torch.autograd.profiler.record_function("forward"):
    with torch.cuda.amp.autocast(enabled=False):
        loss, sample_size, logging_output = criterion(model, sample)
# %%
with torch.autograd.profiler.record_function("backward"):
    optimizer.backward(loss)

# %%
# I got the following from transformer_base.py in models/transformer
# label_smoothed_cross_entopy has the forward: net_output = model(**sample["net_input"])
encoder_output = model.encoder(sample["net_input"]['src_tokens'], src_lengths=sample["net_input"]['src_lengths'], return_all_hiddens=True)
encoder_out = encoder_output["encoder_out"]

# %%
decoder_out = model.decoder(sample["net_input"]['prev_output_tokens'],
        encoder_out=encoder_output,
        features_only=False,
        alignment_layer=4,
        alignment_heads=1,
        src_lengths=sample["net_input"]['src_lengths'],
        return_all_hiddens=True,
    )

# %%
# The following is from transformer_pg.py 
x, extra = model.decoder.extract_features(
            sample["net_input"]['prev_output_tokens'],
            encoder_out=encoder_output,
            incremental_state=None,
            alignment_layer=4,
            alignment_heads=1,
        )
# %%
# Embedding the tokens again for generation probability prediction,
# so that we don't have to reimplement the whole extract_features()
# method.

# if incremental_state is not None:
#     prev_output_tokens = prev_output_tokens[:, -1:]
prev_output_embed = model.decoder.embed_tokens(sample["net_input"]['prev_output_tokens'])
prev_output_embed *= model.decoder.embed_scale
predictors = torch.cat((prev_output_embed, x), 2)
p_gens = model.decoder.project_p_gens(predictors)
p_gens = torch.sigmoid(p_gens.float())
# Torchscript complains if encoder_out or attn are None because
# `output_layer()` signature expects tensors instead
attn = extra["attn"][0]
assert encoder_out is not None
assert attn is not None
x = model.decoder.output_layer(x, attn, encoder_output["src_tokens"][0], p_gens)# %%

# %%
def output_layer(features, attn, src_tokens, p_gens):
    """
    Project features to the vocabulary size and mix with the attention
    distributions.
    """
    if self.force_p_gen is not None:
        p_gens = self.force_p_gen

    # project back to size of vocabulary
    if self.adaptive_softmax is None:
        logits = self.output_projection(features)
    else:
        logits = features

    batch_size = logits.shape[0]
    output_length = logits.shape[1]
    assert logits.shape[2] == self.num_embeddings
    assert src_tokens.shape[0] == batch_size
    src_length = src_tokens.shape[1]

    # The final output distribution will be a mixture of the normal output
    # distribution (softmax of logits) and attention weights.
    gen_dists = self.get_normalized_probs_scriptable(
        (logits, None), log_probs=False, sample=None
    )
    gen_dists = torch.mul(gen_dists, p_gens)
    padding_size = (batch_size, output_length, self.num_oov_types)
    padding = gen_dists.new_zeros(padding_size)
    gen_dists = torch.cat((gen_dists, padding), 2)
    assert gen_dists.shape[2] == self.num_types

    # Scatter attention distributions to distributions over the extended
    # vocabulary in a tensor of shape [batch_size, output_length,
    # vocab_size]. Each attention weight will be written into a location
    # that is for other dimensions the same as in the index tensor, but for
    # the third dimension it's the value of the index tensor (the token ID).
    attn = torch.mul(attn.float(), 1 - p_gens)
    index = src_tokens[:, None, :]
    index = index.expand(batch_size, output_length, src_length)
    attn_dists_size = (batch_size, output_length, self.num_types)
    attn_dists = attn.new_zeros(attn_dists_size)
    attn_dists.scatter_add_(2, index, attn.float())

    # Final distributions, [batch_size, output_length, num_types].
    return gen_dists + attn_dists

# %%
# from criteriton label_smoothed_cross_entropy
def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

def compute_loss(self, model, net_output, sample, reduce=True):
    lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
    loss, nll_loss = label_smoothed_nll_loss(
        lprobs,
        target,
        self.eps,
        ignore_index=self.padding_idx,
        reduce=reduce,
    )
    return loss, nll_loss



# %%
for name, p in model.named_parameters():
    print(name, p.grad.shape)
# %%
for name, p in model.named_parameters():
    print(name, p.grad_sample.shape)

# %%
