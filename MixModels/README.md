## AI@加密流识别算法代码
### 项目背景
STC等海外运营商要求对BIGO、 IMO等VOIP和直播视频软件和VPN应用做封堵（影响运营商利益）。这些应用/VPN的特征不仅被加密还经常发生变异，基于SA库的识别响应需要离线分析，
响应速度以天计算，需要用基于AI持续学习的识别来加速响应。

### 目录说明

##### data_pipeline
流量数据流水线代码，将抓取的pcap包处理成模型训练的样本数据。
##### ms
mindspore版本的算法代码实现。
##### torch
torch版本的算法代码实现。
##### nn_converter
siteai提供的模型格式转换工具，用于将AI中心训练好的ONNX模型转换成适合在前台DPI设备上跑的LITE模型。
##### tools
目前tools用于简化部署模型的生成，并关联相关的训练元数据，方便定位。
### 环境配置

| 依赖 | 版本号 |
| ------ | ------ |
|  Torch      |   1.10.1     |
|  Mindspore  |  Enterprise 103.1.0.B202 (商用版本火车)   |
|  numpy      |   1.18.1  |
|  pandas     |     1.0.3  |
|  scikit-learn   |   0.22.2.post1 |

Torch和Mindspore都属于训练框架，二选一即可。


# VPN Identity Delivery 项目说明

## 目录结构说明
vpn_identify_delivery/
│
├── model/                # 存储模型训练过程中的检查点文件
│   └── model.pth               # 训练得到的模型权重文件
│   └── model.onnx              # 训练得到的模型权重文件，input size基于config/model_config.json中的onnx_saving_batch_size
│
├── config/                     # 存放配置文件的目录
│   └── model_config.json       # 模型配置文件
│   └── results.json            # 重要结果缓存文件，可直接读取并获取结果
│
├── Dataset/                    # 数据集目录
│   ├── Testing_Set/            # 测试数据集目录，一个类别对应一个子文件夹
│   │── Training_Set/           # 训练数据集目录，一个类别对应一个子文件夹
│   ├── appConfig.csv           # 应用程序配置数据
│
├── data.py                     # 数据处理脚本
├── evaluation.py               # 模型评估脚本
├── model.py                    # 模型定义脚本
├── threshold_search.py         # 阈值搜索脚本，搜索结果将存入config/results.json
├── train.py                    # 模型训练脚本
└── utils.py                    # 工具函数脚本，包含获取threshold和outdim的接口函数
