# GCEAN

The code of paper: Unsupervised Ego- and Exo-centric Dense Procedural Activity Captioning via Gaze Consensus Adaptation (ACM MM 2025) is available！

## Requirements

Following the environment of PDVC: https://github.com/ttengwang/PDVC and CM2 https://github.com/ailab-kyunghee/CM2_DVC

## PDVC

Ego2Exo

```
cd PDVC/
bash szf_scripts_new/ego2exo.sh
```

Exo2Ego

```
cd PDVC/
bash szf_scripts_new/exo2ego.sh
```

## CM2

Ego2Exo

```
cd CM2/
bash szf_scripts_new/ego2exo.sh
```

Exo2Ego

```
cd CM2/
bash szf_scripts_new/exo2ego.sh
```

## Data
The EgoMe-UE2DPAC benchmark is constructed based on the EgoMe dataset https://huggingface.co/datasets/HeqianQiu/EgoMe

You can access the processed data of our proposed EgoMe-UE2DPAC benchmark by contacting the author by e-mail: zfshi@std.uestc.edu.cn

## References

**GCEAN:**

```
@inproceedings{shi2025unsupervised,
  title={Unsupervised Ego-and Exo-centric Dense Procedural Activity Captioning via Gaze Consensus Adaptation},
  author={Shi, Zhaofeng and Qiu, Heqian and Wang, Lanxiao and Wu, Qingbo and Meng, Fanman and Li, Hongliang},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={3731--3740},
  year={2025}
}
```

**EgoMe dataset:**

```
@article{qiu2025egome,
  title={Egome: Follow me via egocentric view in real world},
  author={Qiu, Heqian and Shi, Zhaofeng and Wang, Lanxiao and Xiong, Huiyu and Li, Xiang and Li, Hongliang},
  journal={arXiv e-prints},
  pages={arXiv--2501},
  year={2025}
}
```




