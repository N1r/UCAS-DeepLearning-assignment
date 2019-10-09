# 深度学习实验十：神经机器翻译

---

### 运行方式：
1. 在当前目录下  ```git clone https://github.com/tensorflow/nmt/``` 也可以下载完把nmt文件夹放到当前文件夹下
2. data（数据集）文件夹也放到当前文件夹下
3. ```mkdir model```   （新建一个文件夹）
4. 训练
```
python -m nmt.nmt.nmt  
    --attention=scaled_luong 
    --src=zh --tgt=en 
    --vocab_prefix=./data/vocab  
    --train_prefix=./data/train 
    --dev_prefix=./data/dev  
    --test_prefix=./data/test 
    --out_dir=./model 
    --num_train_steps=12000 
    --steps_per_stats=100 
    --num_layers=2 
    --num_units=128 
    --dropout=0.2 
    --metrics=bleu
```
5. 翻译指定文档test.zh
```
python -m nmt.nmt.nmt \
    --out_dir=./model \
    --inference_input_file=./data/test.zh \
    --inference_output_file=./翻译结果.txt
```
