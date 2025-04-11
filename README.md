# Speech to Text API

这是一个基于 Flask 和 Vosk 的语音转文本 API 服务，集成了 DeepSeek API 功能。

## 功能特点

- 支持多种音频格式（自动转换为 WAV 格式）
- 使用 Vosk 离线语音识别模型
- 提供 RESTful API 接口
- 支持跨域请求
- 集成 DeepSeek API 的聊天和文本嵌入功能

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境变量配置

在使用 DeepSeek API 之前，需要设置 API 密钥：

```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

## 下载语音模型

在使用之前，需要下载 Vosk 语音识别模型。模型会自动下载到 `model` 目录下。

## 运行服务

```bash
python app.py
```

服务将在 http://localhost:5000 启动。

## API 使用说明

### 语音转文本

**请求：**

- 方法：POST
- 路径：/transcribe
- 参数：file（音频文件）

**响应：**

```json
{
  "text": "识别出的文本内容",
  "processing_time": "处理时间（秒）"
}
```

### DeepSeek 聊天

**请求：**

- 方法：POST
- 路径：/chat
- 参数：

```json
{
  "messages": [
    { "role": "user", "content": "你的问题" },
    { "role": "assistant", "content": "AI的回答" }
  ],
  "model": "deepseek-chat",
  "temperature": 0.7
}
```

### 文本嵌入

**请求：**

- 方法：POST
- 路径：/embedding
- 参数：

```json
{
  "text": "要生成嵌入的文本",
  "model": "deepseek-embedding"
}
```

## 注意事项

- 确保有足够的磁盘空间用于存储语音模型
- 建议使用 16kHz 采样率的音频文件以获得最佳识别效果
- 首次运行时会自动下载语音模型，可能需要一些时间
- 使用 DeepSeek API 需要有效的 API 密钥
