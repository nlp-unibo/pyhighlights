# Codebase schema

## pyhighlights/components/models/spp/genspp.py

The module depends on `pyhighlights.components.models`, `pyhighlights.components.models.spp.base`, `pyhighlights.utility.diagnostics`, `cinnamon.registry`, and `torch`.
`pyhighlights.components.tasks` calls `GenSPPTrainer.fit` from `GenSPPTask.train`.

`GenSPP` extends `SPP` and overrides `select`, `generator_parameters`, `predictor_parameters`, `on_train_epoch_start`, and `configure_optimizers`.

`GenSPPTrainer.fit` calls `_fit`.
`_fit` calls `_Rows.of`, `_candidate`, `_count_threshold_genes`, `_open_pool`, `_search`, and `_close_pool`.
`_Rows.of` depends on `torch.utils.data.DataLoader` and calls the loader's `collate_fn` once over the whole training split.
`_search` calls `_founder_chromosome`, `_score`, and `_run_generation`.
`_run_generation` calls `_crossover`, `_mutate`, `_score`, and `_select_survivors`.

`_score` calls `_evaluate_individual` directly for one device or under diagnostics.
It calls `_evaluate_individual` through a thread pool for CUDA devices, and through `_score_in_worker` in the process pool for several CPU devices.
It then calls `_restored` for each new best candidate.
`_open_pool` creates the process pool with `_start_worker` as initializer and submits `_worker_ready` once per device.
`_score_in_worker` calls `_evaluate_individual` and `_trained_state` on the trainer that `_start_worker` installed.
Work items, results, and initializer data cross the process boundary as `pickle` bytes.

`_evaluate_individual` calls `_candidate`, `_train_predictor`, `_evaluate`, `compute_fitness`, and `_chromosome`.
`_train_predictor` calls `_Rows.batches` once per epoch.
`_candidate` calls `Registry.from_key`, `SPP.load_embeddings`, and `_align_initial_state`.
`_align_initial_state` calls `_generator_names`.
`_restored` calls `_candidate`.
