# WResVLM

Implementation of **WResVLM**, from the following paper:

Towards Real-World Adverse Weather Image Restoration: Enhancing Clearness and Semantics with Vision-Language Models (ECCV 2024)

Jiaqi Xu, Mengyang Wu, Xiaowei Hu, Chi-Wing Fu, Qi Dou, Pheng-Ann Heng

<p align="center">
<img src="./assets/overview.png"
    class="center">
</p>

## Datasets

We use several (pseudo-)synthetic datasets, including [Outdoor-Rain](https://github.com/liruoteng/HeavyRainRemoval), [RainDrop](https://github.com/rui1996/DeRaindrop), [SPA](https://github.com/zhuyr97/WGWS-Net), [OTS](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2), [Snow100K](https://sites.google.com/view/yunfuliu/desnownet).
Meanwhile, we use real-world data from [URHI](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2) and our collected real rain and snow images for model training.
The real rain and snow images can be downloaded [here](https://appsrv.cse.cuhk.edu.hk/~jqxu/data/WResVLM/WReal.zip).

## License

This project is released under the [MIT license](./LICENSE).
Parts of this project use code, data, and models from other sources, which are subject to their respective licenses.
