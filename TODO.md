# TODO List

## Features
- [x] Divide npy data into data and history in the dataloader
- [x] Update `model_tf.py` and `sample.py` to handle history input
- [x] Implement the history embedder
- [x] Update the sampling script with padding
- [x] Correct the mask generation for MHA (Multi-Head Attention)
- [x] Generate masks for both agent and sequence length padding
- [x] Train a single-agent model on nuScenes
- [x] Create new train/val nuScenes data by normalizing and replacing padded values with previous/next non-zero values
- [x] Train a multi-agent (VEH) model on normalized and saturated nuScenes data
- [x] Implement autoencoder architecture for map compression
- [x] Add agent attention mechanism
- [x] Add padding to the scaled dataset (random padding of total size 3, in positions: begin, middle, end, begin_end)
- [x] Add masked transformer to handle padded sequence length (refer to Masked Attention in Wayformer)
- [x] Include padded agents in the masked transformer
- [x] Tune architecture in `main` with `model_td_no_mask_best.py` and `train_nag_hist.py`
- [x] Experiment with modifying the last mask usage (e.g., first only, last only)
- [x] Train a fully padded model (agent and sequence length) with the best configuration
- [x] Preprocess the entire dataset
- [x] Implement evaluation script
- [ ] Preprocess the map features
- [ ] Add conditioning on the map
- [ ] Tune architecture and report results with all metrics

## Bugs
- [x] Fix bugs related to the `HistoryEmbedder`

## Enhancements
- [ ] Improve generation quality with agent attention
- [ ] Enhance generation using classifier-free guidance based on agent type

## Conclusions
- [Yes] Does conditioning on the final layer enhance generation?
- [Yes] Is concatenating at the beginning of `x` and `h` important for continuity between history and predicted trajectories?
- [Yes] Does using `HistEmbedder` enhance conditional generation?
- [No] Can we use a timestep of shape `(B*N,)` from the beginning?
- [No] Does using PE (Positional Encoding) initialization enhance generation? (No significant difference)
- [No] What if we add the history to the input and condition on `t_em` only? (They don't have the same sequence length)
- [No] Does adding the reshaped `h` to the condition without any history embedding enhance generation (`model_td_no_mask_test`)? 
- [Yes] Does varying model size improve generation? (Start with a larger model size to achieve better generation)
- [Yes] Does increasing the history length improve generation?
- [No] Does `AdaTransformer` in `HistEmbedder` enhance conditional generation? (It performs worse!)
- [Yes] Does using the mask only in the final layer enhance generation?
- [No] Does using the second type of masking (mask at the beginning and end, with mask from `h` referring to padded agents) improve results?
- [No] Should we apply the mask?
- [ ] Does agent attention enhance trajectory generation with respect to each agent?
- [ ] Does map conditioning enhance trajectory generation?
