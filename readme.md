# 🦆  Reproduce "Attacking Optical Flow"

## Results
All models were trained on FlyingChairs dataset. A set of 1000 test images was also selected from FlyingChairs. Patch size is 50x50 pixels.

| Model Name | EPE | EPE (random) | Rel EPE (random) | EPE (adversarial) | Rel EPE (adversarial) | Time |
| ---------- | --- | ------------ | ---------------- | ----------------- | --------------------- | ---- |
| flownet | 1.7576 | 1.9098 | 8.6607  | 1.8816 | 7.0557  | 55s      |
| pwc     | 1.4578 | 1.6004 | 9.7885  | 1.5588 | 6.9338  | 1m 16s   |
| raft    | 0.7699 | 0.8486 | 10.2149 | 0.8717 | 13.2205 | 3m 2s    |

## TODO
- [x] select and download models with checkpoints (flownetc, pwc, raft)
- [x] how to download and add adversarial patch
- [x] create repo
- [x] test EPE from mmflow (https://github.com/open-mmlab/mmflow/blob/master/mmflow/core/evaluation/metrics.py)
- [ ] add how to run
- [ ] visualize diffreence between predicted uattacked and patched input
- [ ] metrics all, without patch, inside patch only
- [ ] average runs with different seeds
- [ ] add automatic result table generation
- [ ] check if model weights are identical
- [ ] universal patch was not optimised for raft

