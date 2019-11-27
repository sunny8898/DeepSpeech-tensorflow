# deepspeech 环境搭建
新建虚拟环境：conda create -n tensorflow python=3.6
激活虚拟环境：source activate tensorflow
1.安装tensorflow：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.13.1
2.安装keras：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple keras==2.2.4
3.安装语音流处理模块：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple soundfile==0.10.2
训练环境安装前三个就可以，测试环境需要后面两个
4.安装beam search解码模块：pip install https://index.taskcluster.net/v1/task/project.deepspeech.deepspeech.native_client.v0.5.0-alpha.11.cpu-ctc/artifacts/public/ds_ctcdecoder-0.5.0a11-cp36-cp36m-manylinux1_x86_64.whl
报错platform不支持的话在DeepSpeech里面执行进行安装：pip install $(python util/taskcluster.py --decoder)
gpu版：pip install https://index.taskcluster.net/v1/task/project.deepspeech.deepspeech.native_client.v0.5.0-alpha.11.cpu-ctc/artifacts/public/ds_ctcdecoder-0.5.0a11-cp35-cp35m-manylinux1_x86_64.whl
5.读字节流：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pydub

# 模型部署
./speech_model里面放入训练好的pb模型，和训练集的std、mean数据
./LM_model里面放入训练好的n-gram语言模型
入口voice_to_text.py




