# 基于 Adversarial Attack 的问题等价性判别比赛

基于 BERT 的 Baseline 模型

环境采用:
+ Python 3.68
+ PyTorch 1.3
+ Transformers

生成数据集
```bash
python preprocessor.py --data_path ./data
```

训练数据，查看全部的参数可以使用 `-h` 参数
```
python main.py
```

生成结果，xxx指某个具体的步数，可以查看输出文件夹中保存的模型名称
```
python main.py --mode test --best_step xxxx
```