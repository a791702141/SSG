# Self-Supervised Graph Neural Network for Multi-Source Domain Adaptation
This project is the official implementation of ``Self-Supervised Graph Neural Network for Multi-Source Domain Adaptation'' in PyTorch, which is accepted by ACM MM 2022.

### Prerequisites

* Python 3.6
* PyTorch 1.4.0  
* CUDA 9.0 & cuDNN 7.0.5

### Dataset Preparation

* [Office-Caltech](https://drive.google.com/file/d/1Q-ABkNTmw4bMJMKLsDZ0h0WtGvzlzhNc/view?usp=sharing)
* [Office-31](http://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
* [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)
* [DomainNet](http://ai.bu.edu/M3SDA/)

### Pre-trained Models


### Training

To train the full model of SSG, simply run:
```
python train.py --use_target --save_model --target clipart \
                --checkpoint_dir $save_dir$
```

Like
```
python train.py --use_target --save_model --target clipart
```

A large body of the code is borrowed from "Learning to Combine: Knowledge Aggregation for Multi-Source Domain Adaptation". Thanks!


## Citation

If this work helps your research, please cite the following paper:
```

@inproceedings{yuan2022self,
  title={Self-Supervised Graph Neural Network for Multi-Source Domain Adaptation},
  author={Yuan, Jin and Hou, Feng and Du, Yangzhou and Shi, Zhongchao and Geng, Xin and Fan, Jianping and Rui, Yong},
  booktitle={Proceedings of the 30th ACM international conference on multimedia},
  year={2022}
}
```

