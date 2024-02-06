## 说明
寒假练手产物，将GCNII主要组件复现了一下，同时复现了可由该模型退化至的模型APPNP，目前仅对半监督部分进行了实现。    
- 基于系统：Arch Linux 6.7.3-arch1-2
- CPU：Ryzen 9 7940H w/ Radeon 780M Graphics (16)
- GPU：GeForce RTX 4060 Max-Q / Mobile
## 快速使用
`git clone`该文件夹并进入，执行
### semi
```zsh
python semi_train/train.py --model GCNII --dataset cora --test

>>> Model GCNII , loading epoch 0540...
>>> best loss.:0.848, acc.:82.400,total time.:33.51
>>> Test acc.:85.800
``` 
### full
```zsh
python full_train/train.py --data cora --nlayer 64 --alpha 0.2 --wd1 1e-4 --wd2 1e-4

>>> total time.:335.91
>>> Val acc(mean).:88.568
```
