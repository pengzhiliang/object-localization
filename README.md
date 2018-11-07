# Object localization and detection

1. 数据集：
* [tiny vid](http://xinggangw.info/data/tiny_vid.zip)
* 每类共180张图片，前150张train,后30张test，一共5类

2. data augmentation
* 色彩变化
* 水平反转
* crop & resize

3. 两种实现
* simple net 
```bash
python eval.py
```
* ssd
```bash
python SSD_eval.py
```
4. reference
[pytorch-ssd300](https://github.com/kuangliu/torchcv)
