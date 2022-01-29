# Reproduce "Attacking Optical Flow"

## Results
| Model Name | EPE | EPE with random patch | Rel EPE (random) | EPE (adversarial) | Rel EPE (adversarial) | Time |
| ---------- | --- | --------------------- | ---------------- | ----------------- | --------------------- | ---- |
| flownet | 1.7085 | 1.8315 | 7.1978 | 1.8168 | 6.3381  | 5s  |
| pwc     | 1.4238 | 1.5352 | 7.8187 | 1.5142 | 6.3496  | 7s  |
| raft    | 0.7196 | 0.7196 | 6.8983 | 0.8158 | 13.3641 | 18s |

## TODO
- [x] select and download models with checkpoints (flownetc, pwc, raft)
- [x] how to download and add patch
- [x] create repo
- [x] test EPE from mmflow (https://github.com/open-mmlab/mmflow/blob/master/mmflow/core/evaluation/metrics.py)
- [ ] add how to run
- [ ] visualize diffreence between predicted uattacked and patched input
- [ ] metrics all, without patch, inside patch only
- [ ] average runs with different seeds

