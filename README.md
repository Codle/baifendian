# 基于 Adversarial Attack 的问题等价性判别

采用 BERT 作为基础模型，实现了 ACL 2019 论文《Combating Adversarial Misspellings with Robust Word Recognition》来防御扰动攻击。

在 RTX 2080 上训练 5 小时，迭代 40k，batch 设置为 8 的结果为：
+ 训练集得分：0.97
+ 验证集得分：0.89
+ 测试集得分：

环境采用:
+ Python 3.68
+ PyTorch 1.3
+ Transformers

生成数据集，其中 `./data` 为数据集文件夹，其中应该包含 `train_set.xml` 和 `dev_set.csv`
```bash
python preprocessor.py --data_path ./data
```

训练数据，查看全部的参数可以使用 `-h` 参数
```bash
python main.py
```

生成结果，2000 指某个具体的步数，可以查看输出文件夹中保存的模型名称确定
```bash
python main.py --mode test --best_step 2000
```