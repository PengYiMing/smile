# smile
enhancement for OpenCV smile detector

>1. 收集数据
放入origin data
>2. 随机选择train/test样本
放入data/train
放入data/test
>3. 对样本进行augmentation
data/pre-process.py
>4. 训练model并保存
cnn/model.py
>5. 从摄像头取帧并加载model进行笑脸识别
detector/detector.py

