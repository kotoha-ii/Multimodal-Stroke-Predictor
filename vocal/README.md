# 脑卒中早筛语音识别系统

## 项目概述
基于语音分析的脑卒中早期筛查系统，通过分析语音特征评估脑卒中风险。

## 主要功能
- 语音上传和分析
- 三类特征提取：
  - 基础声学特征（MFCC、基频）
  - 高级声学特征（jitter、shimmer、谐波比）
  - 语言学特征（语速、停顿频率）
- 生成中文分析报告

## 项目结构
```
├── app.py                # Flask主应用
├── core/
│   └── audio_processor.py # 音频处理核心逻辑
├── uploads/              # 上传音频存储
├── web/
│   ├── static/js/        # 前端JavaScript
│   └── templates/
│       └── index.html    # 主页面
└── venv/                 # Python虚拟环境
```

## 特征介绍

1.  **谐波失真（Jitter）**：

含义：测量相邻声波周期之间的频率变化

正常范围：通常在0.1%到1%之间

临床意义：过高的Jitter可能表明声带振动不稳定



2. **基频稳定性（F0标准差）**：

含义：说话时音高变化的程度

正常范围：通常在10-50 Hz之间

临床意义：过高的标准差可能表明声音控制能力下降

## 安装与运行

1. 克隆项目：

```
git clone https://github.com/kotoha-ii/Multimodal-Stroke-Predictor.git
cd stroke_speech_system
```



2. 创建虚拟环境：

```
python -m venv venv
```



3. 激活虚拟环境：

Windows:

```
venv\Scripts\activate
```

Linux/Mac:

```
source venv/bin/activate
```



4. 安装依赖：

```
pip install -r requirements.txt
```



5. 运行应用：

```
python app.py
```

## 技术栈
- 后端：Python (Flask, librosa, pyAudioAnalysis)
- 前端：HTML/JavaScript

## 使用说明
1. 访问应用首页
2. 上传语音文件
3. 查看分析结果

## 注意事项
1. 确保上传的音频文件格式为常见格式（如.wav, .mp3）
2. 建议录音环境保持安静，以获得更准确的分析结果
3. 系统分析结果仅供参考，不能替代专业医疗诊断