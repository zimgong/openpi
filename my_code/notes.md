<!-- convert pi0 jax to lerobot -->
python lerobot/common/policies/pi0/conversion_scripts/convert_pi0_to_hf_lerobot.py \
    --checkpoint_dir /home/remi_cadene/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim/params \
    --output_path /home/remi_cadene/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch

TODO: remember to export PYTHONPATH before running

export PYTHONPATH=/data/ceph_hdd/main/dev/shaoze.yang/code_ly:$PYTHONPATH

python lerobot/common/policies/pi0/conversion_scripts/convert_pi0_to_hf_lerobot.py \
    --checkpoint_dir ../openpi/checkpoints/pi0_robocasa_v0.1/pi0_0330/25000/params \
    --output_path ../openpi/checkpoints/pi0_0330_25000_pytorch


<!-- stats -->
Writing stats to: /data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/assets/pi0_robocasa/robocasa/data-collection-3000

<!-- video dimension -->
224, 224, 3

<!-- finetune -->
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_robocasa --exp-name=my_experiment

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_fast_robocasa --exp-name=test_1

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_robocasa --exp-name=pi0

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_robocasa --exp-name=pi0 --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_robocasa --exp-name=pi0 --resume

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_robocasa_v0.1 --exp-name=pi0_0330 --resume

<!-- pi0 work space-->
cd /data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi

<!-- environment -->
source .venv/bin/activate

<!-- Example downloading parameters: -->
```bash
python
>>> import openpi.shared.download as download
>>> path='s3://openpi-assets/checkpoints/pi0_base/params'
>>> download.maybe_download(path)
```

<!-- run stats before fintune -->
python scripts/compute_norm_stats.py --config-name pi0_fast_robocasa

python scripts/compute_norm_stats.py --config-name pi0_robocasa_v0.1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python scripts/compute_norm_stats.py --config-name pi0_robocasa_v0.1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/compute_norm_stats.py --config-name pi0_robocasa_v0.1

<!-- pi0 finetune checkpoints -->
/data/ceph_hdd/main/dev/zim.gong/lerobot/outputs/train/2025-03-11/11-02-21_pi0/checkpoints


<!-- full size data -->
/data/ceph_hdd/main/dev/zim.gong/lerobot/data/robocasa/data-collection-3000

<!-- convert pi0 to lerobot -->

python lerobot/lerobot/common/policies/pi0/conversion_scripts/convert_pi0_to_hf_lerobot.py \
    --checkpoint_dir /home/remi_cadene/.cache/openpi/openpi-assets/checkpoints/pi0_base/params \
    --output_path /home/remi_cadene/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch



<!-- error to fix -->

(openpi) shaoze.yang@pvevm:/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi$ python scripts/compute_norm_stats.py --config-name pi0_fast_robocasa
Some kwargs in processor config are unused and will not have any effect: scale, min_token, time_horizon, action_dim, vocab_size. 
Some kwargs in processor config are unused and will not have any effect: scale, min_token, time_horizon, action_dim, vocab_size. 
Returning existing local_dir `/home/shaoze.yang/.cache/huggingface/lerobot/robocasa/data-collection-3000` as remote repo cannot be accessed in `snapshot_download` (None).
WARNING:huggingface_hub._snapshot_download:Returning existing local_dir `/home/shaoze.yang/.cache/huggingface/lerobot/robocasa/data-collection-3000` as remote repo cannot be accessed in `snapshot_download` (None).
Traceback (most recent call last):
  File "/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/scripts/compute_norm_stats.py", line 80, in <module>
    tyro.cli(main)
  File "/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/.venv/lib/python3.11/site-packages/tyro/_cli.py", line 189, in cli
    return run_with_args_from_cli()
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/scripts/compute_norm_stats.py", line 44, in main
    data_config, dataset = create_dataset(config)
                           ^^^^^^^^^^^^^^^^^^^^^^
  File "/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/scripts/compute_norm_stats.py", line 27, in create_dataset
    dataset = _data_loader.create_dataset(data_config, config.model)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/src/openpi/training/data_loader.py", line 92, in create_dataset
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, local_files_only=data_config.local_files_only)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/.venv/lib/python3.11/site-packages/lerobot/common/datasets/lerobot_dataset.py", line 88, in __init__
    self.info = load_info(self.root)
                ^^^^^^^^^^^^^^^^^^^^
  File "/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/.venv/lib/python3.11/site-packages/lerobot/common/datasets/utils.py", line 157, in load_info
    info = load_json(local_dir / INFO_PATH)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/.venv/lib/python3.11/site-packages/lerobot/common/datasets/utils.py", line 129, in load_json
    with open(fpath) as f:
         ^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/home/shaoze.yang/.cache/huggingface/lerobot/robocasa/data-collection-3000/meta/info.json'

<!-- solution -->
export LEROBOT_HOME = [raw_data_path] for lerobot 2.0
export HF_LEROBOT_HOME = [raw_data_path] for lerobot 2.1

/data/ceph_hdd/main/dev/zim.gong/lerobot/data




<!-- config -->
config: TrainConfig(name='pi0_fast_robocasa', project_name='openpi', exp_name=<tyro._singleton.PropagatingMissingType object at 0x7efe51727d90>, model=Pi0FASTConfig(action_dim=13, action_horizon=10, max_token_len=180, dtype='bfloat16', paligemma_variant='gemma_2b'), weight_loader=CheckpointWeightLoader(params_path='s3://openpi-assets/checkpoints/pi0_fast_base/params'), lr_schedule=CosineDecaySchedule(warmup_steps=1000, peak_lr=2.5e-05, decay_steps=30000, decay_lr=2.5e-06), optimizer=AdamW(b1=0.9, b2=0.95, eps=1e-08, weight_decay=1e-10, clip_gradient_norm=1.0), ema_decay=0.99, freeze_filter=Nothing(), data=LeRobotRobocasaDataConfig(repo_id='robocasa/data-collection-3000', assets=AssetsConfig(assets_dir=None, asset_id=None), base_config=DataConfig(repo_id=None, asset_id=None, norm_stats=None, repack_transforms=Group(inputs=(), outputs=()), data_transforms=Group(inputs=(), outputs=()), model_transforms=Group(inputs=(), outputs=()), use_quantile_norm=False, action_sequence_keys=('actions',), prompt_from_task=True, local_files_only=True), action_sequence_keys=('action',)), assets_base_dir='./assets', checkpoint_base_dir='./checkpoints', seed=42, batch_size=32, num_workers=2, num_train_steps=30000, log_interval=100, save_interval=1000, keep_period=5000, overwrite=False, resume=False, wandb_enabled=True, policy_metadata=None, fsdp_devices=1)

<!-- batch.keys() -->
dict_keys(['actions', 'image', 'image_mask', 'state'])


<!-- config -->
TrainConfig(name='pi0_robocasa', project_name='openpi', exp_name='pi0', model=Pi0Config(action_dim=32, action_horizon=50, max_token_len=48, dtype='bfloat16', paligemma_variant='gemma_2b', action_expert_variant='gemma_300m'), weight_loader=CheckpointWeightLoader(params_path='s3://openpi-assets/checkpoints/pi0_base/params'), lr_schedule=CosineDecaySchedule(warmup_steps=1000, peak_lr=2.5e-05, decay_steps=30000, decay_lr=2.5e-06), optimizer=AdamW(b1=0.9, b2=0.95, eps=1e-08, weight_decay=1e-10, clip_gradient_norm=1.0), ema_decay=0.99, freeze_filter=Nothing(), data=LeRobotRobocasaDataConfig(repo_id='robocasa/data-collection-3000', assets=AssetsConfig(assets_dir=None, asset_id=None), base_config=DataConfig(repo_id=None, asset_id=None, norm_stats=None, repack_transforms=Group(inputs=(), outputs=()), data_transforms=Group(inputs=(), outputs=()), model_transforms=Group(inputs=(), outputs=()), use_quantile_norm=False, action_sequence_keys=('actions',), prompt_from_task=True, local_files_only=True), action_sequence_keys=('action',)), assets_base_dir='./assets', checkpoint_base_dir='./checkpoints', seed=42, batch_size=2, num_workers=2, num_train_steps=30000, log_interval=100, save_interval=1000, keep_period=5000, overwrite=True, resume=False, wandb_enabled=True, policy_metadata=None, fsdp_devices=1)


<!-- error -->
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2700/2700 [00:00<00:00, 320085.38it/s]
21:28:11.769 [I] Initialized data loader:
[0].images['base_0_rgb']: (2, 224, 224, 3)@float32
[0].images['left_wrist_0_rgb']: (2, 224, 224, 3)@float32
[0].images['right_wrist_0_rgb']: (2, 224, 224, 3)@float32
[0].image_masks['base_0_rgb']: (2,)@bool
[0].image_masks['left_wrist_0_rgb']: (2,)@bool
[0].image_masks['right_wrist_0_rgb']: (2,)@bool
[0].state: (2, 32)@float32
[0].tokenized_prompt: (2, 48)@int32
[0].tokenized_prompt_mask: (2, 48)@bool
[1]: (2, 50, 32)@float32 (957247:train.py:228)
21:28:12.355 [I] Created BasePyTreeCheckpointHandler: pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=None (957247:base_pytree_checkpoint_handler.py:332)
21:28:12.379 [I] Restoring checkpoint from /home/shaoze.yang/.cache/openpi/openpi-assets/checkpoints/pi0_base/params. (957247:checkpointer.py:256)
21:28:44.205 [I] [process=0] /jax/checkpoint/read/bytes_per_sec: 388.1 MiB/s (total bytes: 12.1 GiB) (time elapsed: 31 seconds) (per-host) (957247:base_pytree_checkpoint_handler.py:113)
21:28:44.206 [I] Finished restoring checkpoint from /home/shaoze.yang/.cache/openpi/openpi-assets/checkpoints/pi0_base/params. (957247:checkpointer.py:259)
21:28:44.206 [I] [process=0][thread=MainThread] Skipping global process sync, barrier name: Checkpointer:restore (957247:multihost.py:293)
2025-03-14 21:28:48.476592: W external/xla/xla/hlo/transforms/simplifiers/hlo_rematerialization.cc:3021] Can't reduce memory use below -25.62GiB (-27506732520 bytes) by rematerialization; only reduced to 48.25GiB (51808779708 bytes), down from 48.25GiB (51808779708 bytes) originally
2025-03-14 21:29:02.854478: W external/xla/xla/tsl/framework/bfc_allocator.cc:501] Allocator (GPU_0_bfc) ran out of memory trying to allocate 4.50GiB (rounded to 4831838208)requested by op 
2025-03-14 21:29:02.854640: W external/xla/xla/tsl/framework/bfc_allocator.cc:512] ***************************************************************************************************_
E0314 21:29:02.854727  957629 pjrt_stream_executor_client.cc:3045] Execution of replica 0 failed: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 4831838208 bytes. [tf-allocator-allocation-error='']
2025-03-14 21:29:02.887840: W external/xla/xla/tsl/framework/bfc_allocator.cc:501] Allocator (GPU_1_bfc) ran out of memory trying to allocate 4.50GiB (rounded to 4831838208)requested by op 
2025-03-14 21:29:02.888121: W external/xla/xla/tsl/framework/bfc_allocator.cc:512] ***************************************************************************************************_
E0314 21:29:02.888213  957632 pjrt_stream_executor_client.cc:3045] Execution of replica 0 failed: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 4831838208 bytes. [tf-allocator-allocation-error='']
Traceback (most recent call last):
  File "/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/scripts/train.py", line 274, in <module>
    main(_config.cli())
  File "/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/scripts/train.py", line 230, in main
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/.venv/lib/python3.11/site-packages/jaxtyping/_decorator.py", line 559, in wrapped_fn
    return wrapped_fn_impl(args, kwargs, bound, memos)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/.venv/lib/python3.11/site-packages/jaxtyping/_decorator.py", line 483, in wrapped_fn_impl
    out = fn(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^
  File "/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/scripts/train.py", line 125, in init_train_state
    train_state = jax.jit(
                  ^^^^^^^^
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 4831838208 bytes.: while running replica 0 and partition 0 of a replicated computation (other replicas may have failed as well).
--------------------
For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.