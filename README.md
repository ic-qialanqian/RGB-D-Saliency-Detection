# RGB-D-Saliency-Detection


## Usage
This is the official code of our paper "[Progressive multi-scale fusion network for RGB-D salient object detection](https://www.sciencedirect.com/science/article/pii/S1077314222001126)".


### 1. Train

1. Set the `--datasets_root_train` and `--datasets_root_test` path in config.py.

2. Run train.py.

3. After training the result model will be stored under `model*` folder.

### 2. Test


```shell
python generate_salmap.py --exp_name='./'
```



### 3. Evaluation results

Run python test_metric_score.py

## BibTex
To cite this code for publications - please use:
```
@article{ren2022progressive,
  title={Progressive multi-scale fusion network for RGB-D salient object detection},
  author={Ren, Guangyu and Xie, Yanchun and Dai, Tianhong and Stathaki, Tania},
  journal={Computer Vision and Image Understanding},
  volume={223},
  pages={103529},
  year={2022},
  publisher={Elsevier}
}

