# Pytorch-PLATO
**PLATO dialog model with pre-trained parameters in pytorch version**

This repository contains:
1. PLATO source code converted in pytorch version.
2. PLATO pre-trained parameters converted in pytorch version: 12-layers, 768-hidden, 12-heads, 132M parameters (uncased model with latent variables).

For simplicity, you can also download the model file `PLATO.pt` from this [link](https://drive.google.com/file/d/1JqO0ZYN0qJmCWVui8jfe6dyyziJfqyo3/view?usp=sharing).

## Requirements
```
- python >= 3.6
- pytorch == 1.8.0
- numpy
- nltk
- tqdm
- regex
```

## Data preparation
Download data from the [link](https://baidu-nlp.bj.bcebos.com/PLATO/data.tar.gz).
The tar file contains three processed datasets: `DailyDialog`, `PersonaChat` and `DSTC7_AVSD`.
```bash
mv /path/to/data.tar.gz .
tar xzf data.tar.gz
```

## Citation
**PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable**
[paper link](https://www.aclweb.org/anthology/2020.acl-main.9.pdf)

Original code link in paddlepaddle version: [https://github.com/PaddlePaddle/Research/tree/master/NLP/Dialogue-PLATO](https://github.com/PaddlePaddle/Research/tree/master/NLP/Dialogue-PLATO)

Other details can also refer to the descriptions in paddlepaddle version from the above link, we keep the same with it!
```
@inproceedings{bao2019plato,
    title={PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable},
    author={Bao, Siqi and He, Huang and Wang, Fan and Wu, Hua and Wang, Haifeng},
    booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
    pages={85--96},
    year={2020}
}
```
