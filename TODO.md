
# TODO List

## Features
- [OK] divide npy data into data and hist in the dataloader
- [OK] update model_tf.py and sample.py to take hist
- [OK] implement the history embedder
- [OK] update the sampling script with pad
- [OK] correct the mask generation for MHA
- [OK] generate the mask for both agent and sequence_length padding
- [ ] device needed in torch.full when generating -inf, 0 mask
- [ ] compare with unitraj data processing 

## Bugs
- [OK] fix HistoryEmbedder related pugs
- [ ] nan loss when using new mask


## Enhancements
- [ ] enhance the collate_fn to take args max_agents and hist_length 
- [ ] Improve with classifier free guidance on agent type