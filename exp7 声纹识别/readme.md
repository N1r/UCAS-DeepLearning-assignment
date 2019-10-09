#深度学习实验七 声纹识别

---

###运行方式：
1. 预处理数据 ```python data_preprocess.py```
2. 模型训练 ```python main.py --train True --iteration 10000  --N 16  --M 5 --model_path ./model```
3. 模型测试 ```python main.py --train False --model_path ./model```

### 参数说明
* '--N'    "number of speakers of batch"
* '--M'    "number of utterances per speaker"

显存大的可以把这两个参数调大 

