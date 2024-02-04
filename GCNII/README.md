## 说明
寒假练手产物，将GCNII主要组件复现了一下，同时复现了可由该模型退化至的模型APPNP，目前仅对半监督部分进行了实现。
## 快速使用
`git clone`该文件夹并进入，执行
```zsh
python semi_train/train.py --model GCNII --dataset cora --test

>>> Model GCNII , loading epoch 0540...
>>> best loss.:0.848, acc.:82.400,total time.:33.51
>>> Test acc.:85.800

``` 
