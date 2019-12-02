

# 和百度deepspeech的不同点
## 1.	框架选择

背景：2019年3月12号智能语音组接受了公司新采购的GPU机器一台，由于新机器适配的驱动版本太高（2019年2月发布），当前语音转写模型使用的深度学习框架theano偏学术研究，theano的开发团队在17年就加入了google，已经停止维护，theano不支持分布式，相比之下tensorflow框架更偏工程，已经是主流框架，支持分布式，支持新硬件，我们有必要对转写工程做框架调整。

百度模型框架：theano_0.8.2、keras_1.1.0

新模型框架：tensorflow_1.13.1、keras_2.2.4

分析：根据调研资料显示，tensorflow新版本相比theano可以带来性能上一倍的提升，同时需要更大的内存。
 
## 2.	声学模型结构
在模型结构上主要做了6项调整，分析了每个调整项带来的影响：

|调整项	| 老模型	| 新模型	| 准确率 | 	性能 | 	资源占用|
|----|----|----|----|----|-----|
|网络结构|	1D_CNN+3*GRU|	1_DCNN+3*BiGRU	|有提升|	降低近一倍|	更大的内存|
|损失函数	|warp-ctc（baidu出品）	|tensorflow-ctc（google出品）	|不确定|	降低一点	|不确定|
|输出节点数|	27|	4563|	有提升	|提升一点|	降低|
|语音帧长|	20ms	|25ms	|有一点提升|	降低1.6倍|	更大的内存|

## 3.其他调整项：

（1）卷积层输出处理：忽略卷积层的前两位输出，因为它们通常无意义，且会影响模型最后的输出；

（2）BN层处理：最后一次训练冻结BN层，传入加载模型（纯开源数据训练的）的移动均值和方差。
调整后准确率平均提升2个百分点

## 4.	增加beam search和n-gram组合解码模块

- 百度模型是贪婪搜索解码
- 新模型的解码模块使用现在GitHub 上比较热门的mozilla基金会实现的beam search解码模型，在权威性、准确率和性能方面都比之前deepspeech好很多，调整后准确率平均提升6个百分点



# deepspeech 环境搭建

新建虚拟环境：conda create -n tensorflow python=3.6

激活虚拟环境：source activate tensorflow

1.安装tensorflow：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.13.1

2.安装keras：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple keras==2.2.4

3.安装语音流处理模块：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple soundfile==0.10.2

训练环境安装前三个就可以，测试环境需要后面两个

4.安装beam search解码模块（解码模块使用mozilla项目里面的）：pip install https://index.taskcluster.net/v1/task/project.deepspeech.deepspeech.native_client.v0.5.0-alpha.11.cpu-ctc/artifacts/public/ds_ctcdecoder-0.5.0a11-cp36-cp36m-manylinux1_x86_64.whl

报错platform不支持的话在mozilla的DeepSpeech里面执行进行安装：pip install $(python util/taskcluster.py --decoder)

gpu版：pip install https://index.taskcluster.net/v1/task/project.deepspeech.deepspeech.native_client.v0.5.0-alpha.11.cpu-ctc/artifacts/public/ds_ctcdecoder-0.5.0a11-cp35-cp35m-manylinux1_x86_64.whl

5.读字节流：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pydub



# 模型部署
./speech_model里面放入训练好的pb模型，和训练集的std、mean数据

./LM_model里面放入训练好的n-gram语言模型

入口voice_to_text.py

# 测试结果
100条数据堂电话语音数据上平均字错率0.02，句错率0.06

详细见./test_result/recongnnize_result.txt

# 说明
master分支是部署代码

train分支是训练代码
