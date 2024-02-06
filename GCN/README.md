## 说明
寒假期间练手复现gcn的产物。只是把主要组件复现了一下。    

- 基于系统：Arch Linux 6.7.3-arch1-2
- CPU：Ryzen 9 7940H w/ Radeon 780M Graphics (16)
- GPU：GeForce RTX 4060 Max-Q / Mobile
## 快速使用
git-clone该文件夹并进入，使用下命令可快速测试。
```zsh
python main.py --dataset=cora  --repeat_num=100
 
>>> Test set results(100 mean): cost= 1.05416 accuracy= 0.80400 time= 191.30230
```

