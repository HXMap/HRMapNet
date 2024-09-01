# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train HRMapNet with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/hrmapnet/hrmapnet_maptrv2_nusc_r50_24ep.py 8
```

HRMapNet should be evaluated with a single GPU for best performance!
```
./tools/dist_test_map.sh ./projects/configs/maptr/hrmapnet_maptrv2_nusc_r50_24ep.py ./path/to/ckpts.pth 1
```
