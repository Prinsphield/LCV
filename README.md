# Learnable Cost Volume Using the Cayley Representation

Taihong Xiao, Jinwei Yuan, Deqing Sun, Qifei Wang, Xin-Yu Zhang, Kehan Xu, Ming-Hsuan Yang

Please cite our paper if you find it useful to your research.
```
@inproceedings{xiao2020learnable,
    title={Learnable cost volume using the cayley representation},
    author={Xiao, Taihong and Yuan, Jinwei and Sun, Deqing and Wang, Qifei and Zhang, Xin-Yu and Xu, Kehan and Yang, Ming-Hsuan},
    booktitle={European Conference on Computer Vision (ECCV)},
    pages={483--499},
    year={2020},
    organization={Springer}
}
```


## Introduction

The learnable cost volume can be easily implemented using either pytorch or tensorflow.

For the pytorch implementation, please refer to [`corr.py`](https://github.com/Prinsphield/LCV/blob/master/corr.py)
This code is modified for RAFT+LCV.
Compare the classes `LearnableCorrBlock` with `CorrBlock` to know the differences between learnable cost volume and vanilla cost volume.

For the tensorflow implementation, please refer to [`network.py`](https://github.com/Prinsphield/LCV/blob/master/network.py#L122-L162) for details.
This code is modified for DDFlow+LCV.


