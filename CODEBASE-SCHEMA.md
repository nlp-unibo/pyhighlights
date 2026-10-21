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

## pyhighlights/components/codes.py

The module depends on `pyhighlights.components.analyzers`, `cinnamon.registry`, `lightning`, `pandas`, and `torch`.
`SelectionCodeAnalyzer` extends `PredictionAnalyzer`; its `analyze` calls `summarize`, its `codes` calls `codes`, its `probes` calls `mask_probes`, its `kept` calls `kept_by_label`, and its `entropy` calls `selection_entropy`.
Its `evaluated` calls `checkpoints`, `_check_corpus`, and `evaluator_readings`.
`checkpoints` calls `replayed` and rebuilds a trained run's task with `Registry.from_key`, then calls its `splits`, `loaders`, and `build_model`.
`replay_predictions` calls `checkpoints` and scores each seed with `lightning.Trainer.test`.
`evaluator_readings` calls the evaluator's `selection_valid` and `predict`.

`selection_entropy` calls `_entropy`.
`mask_probes` calls `mask_features`, `_probeable` and `_probe`, which depends on `scikit-learn` from the `analysis` extra.
`summarize` calls `_rivals`, `complete`, `complete_in_order`, `gold_recall`, `_highlight_scores`, `_where`, `codes`, and `_mean`.
`_rivals` calls `class_highlights`.
`complete` calls `runs` and `contains`.
`complete_in_order` calls `contains`.

## pyhighlights/components/models/spp/evalx.py

The module depends on `pyhighlights.components.models.spp.base` and `torch`.
`EvalX` extends `SPP` and overrides `select`, which calls `RandomSelector.sample` instead of encoding.
Prediction goes through the inherited `SPP.predict`.
`pyhighlights/configurations/evalx.py` registers `RandomSelector`, `EvalX`, and the toy evaluator task.
`pyhighlights.components.codes.evaluator_readings` reads selections with a trained `EvalX`.

## pyhighlights/components/models/spp/grat.py

The module depends on `pyhighlights.components.models`, `pyhighlights.components.models.spp.base`, `pyhighlights.utility.diagnostics`, `pyhighlights.utility.losses`, `cinnamon.registry`, and `torch`.
`GRAT` extends `SPP` and builds a `GRATGuider` with `Registry.from_key`.
`GRAT.load_embeddings` calls `SPP.load_embeddings` and then `GRATGuider.load_embeddings`.
`AttentionGuider` extends `GRATGuider`; its `forward` calls `SPPBackbone.encode`, and its `load_embeddings` calls `SPPBackbone.load_embeddings`.
`pyhighlights.components.tasks` calls `GRAT.load_embeddings` from `SPPTask.build_model`.
