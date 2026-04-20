# Banana Image Generation API - ComfyUI 节点

这是一个支持多个第三方 API 提供商的 Banana 图像生成 ComfyUI 节点，同时包含 Gemini Vision API 节点用于图像和视频分析。

## 功能特性

### Banana Image Generation API
- 支持多个第三方 API 提供商（可扩展）
- 支持多张图片输入（最多 5 张，可选）
- 支持纯文本 prompt 生成
- 可配置图片尺寸和宽高比
- 海外和国内 Host 快速切换
- 自动轮询获取结果
- 可选自动保存到 output/banana 目录
- 详细日志输出（带 emoji 图标）

### Gemini Vision API
- 支持图片输入分析
- 支持视频 URL 输入分析
- 使用 Chat Completions 格式调用 Gemini 模型
- 支持 temperature 和 max_tokens 参数调节
- 可选保存响应文本到文件

## 安装

1. 将此文件夹放置在 ComfyUI 的 `custom_nodes` 目录下
2. 复制 `config.json.example` 为 `config.json`
3. 编辑 `config.json` 填入你的 API Key
4. 重启 ComfyUI

## 配置文件

在 `Banana_Hub_Api` 目录下创建 `config.json` 文件：

```json
{
  "api_keys": {
    "grsai_api": "your_grsai_api_key_here",
    "bltai_api": "your_bltai_api_key_here",
    "147ai_api": "your_147ai_api_key_here"
  }
}
```

配置说明：
- `api_keys`: 存储各个 API 提供商的密钥
  - 键名必须对应 `api/` 目录下的配置文件名（不含 .json 后缀）
  - 例如：`grsai_api` 对应 `api/grsai_api.json`
  - 切换提供商时会自动使用对应的 key

### Gemini Vision API 配置说明

Gemini Vision API 节点使用 `bltai_api` 的 API Key，请确保在 `config.json` 中配置了：

```json
{
  "api_keys": {
    "bltai_api": "your_bltai_api_key_here"
  }
}
```

### 添加更多 API 提供商

如果你添加了新的 API 提供商配置文件（如 `api/another_provider.json`），只需在 `api_keys` 中添加对应的 key：

```json
{
  "api_keys": {
    "grsai_api": "sk-grsai-xxxxx",
    "another_provider": "sk-another-xxxxx"
  }
}
```

节点会根据你选择的 `api_provider` 参数自动使用对应的 API Key。

## API 提供商配置

API 提供商配置文件位于 `api/` 目录下，每个提供商一个 JSON 文件。

### 当前支持的提供商

- **grsai_api**: Grsai 官方 API
  - 国内 Host: https://grsai.dakka.com.cn
  - 海外 Host: https://grsaiapi.com
  - 支持异步轮询模式

- **bltai_api**: BltAI 中转站 (OpenAI Dall-e 格式)
  - 国内 Host: https://api.bltcy.ai
  - 海外 Host: https://api.gptbest.vip
  - 支持同步返回结果
  - 模型: gemini-3.1-flash-image-preview, nano-banana-2, nano-banana-pro

- **147ai_api**: 147AI API
  - 支持 Gemini 原生格式
  - 支持图片生成

- **gemini_api**: Gemini Vision API (用于 Gemini Vision 节点)
  - 使用 bltai_api 的 API Key
  - 模型: gemini-3.1-flash-lite-preview
  - 支持图片和视频分析

### 添加新的 API 提供商

在 `api/` 目录下创建新的 JSON 配置文件，参考 `grsai_api.json` 的格式。重启 ComfyUI 后，新的提供商会自动加载。

---

## 节点说明

### 1. Banana Image Generation API 节点

用于生成图片的节点。

#### 节点参数

- **api_provider**: API 提供商选择
- **host_type**: Host 类型（china/overseas/custom）
- **image1 ~ image5**: 5 个图片输入端口（可选）
- **prompt**: 提示词（必填）
- **model**: 模型选择
- **aspect_ratio**: 宽高比
- **image_size**: 输出图片尺寸
- **seed**: 随机种子
- **timeout**: 请求超时时间（秒）
- **max_retries**: 最大重试次数
- **poll_interval**: 轮询间隔（秒）
- **save_to_output**: 是否保存到 output/banana 目录

#### 输出

- **result_image**: 生成的结果图片
- **log**: 详细的处理日志（带 emoji 图标）

---

### 2. Gemini Vision API 节点

用于分析图片和视频的节点，返回文本分析结果。

#### 节点参数

- **image**: 图片输入（可选）
- **video_url**: 视频 URL（可选，支持 mp4 等格式）
- **prompt**: 提示词/问题（例如："描述这张图片的内容"）
- **host_type**: Host 类型（china/overseas/custom）
  - china: https://api.bltcy.ai
  - overseas: https://api.gptbest.vip
- **temperature**: 采样温度（0.0-2.0，默认 0.7）
- **max_tokens**: 最大生成 token 数（默认 4000）
- **timeout**: 请求超时时间（秒，默认 60）
- **save_response**: 是否保存响应文本到文件

#### 输出

- **response_text**: API 返回的文本分析结果
- **log**: 详细的处理日志（带 emoji 图标）

#### 使用场景

**场景 1：分析图片**
1. 连接一张图片到 `image` 端口
2. 填写 prompt，例如："描述这张图片的内容"
3. 运行节点
4. 从 `response_text` 获取分析结果

**场景 2：分析视频**
1. 在 `video_url` 中填入视频链接（如：https://example.com/video.mp4）
2. 填写 prompt，例如："这个视频讲了什么？"
3. 运行节点
4. 从 `response_text` 获取分析结果

**场景 3：同时分析图片和视频**
1. 同时连接图片和填写视频 URL
2. 填写 prompt
3. 运行节点

#### 注意事项

- 必须至少提供图片或视频 URL 之一
- 使用 `bltai_api` 的 API Key，请确保已配置
- 视频分析需要模型支持视频理解能力
- 响应文本较长时建议启用 `save_response`

---

## 参数显示说明

为了方便识别哪些参数属于哪个提供商，节点会在参数后面标注支持的提供商：

**示例：**
- `nano-banana-fast (Grsai API)` - 只有 Grsai API 支持
- `1:1 (Grsai API, Custom API)` - 两个提供商都支持
- `auto (All)` - 所有提供商都支持

**使用建议：**
- 选择带有当前提供商名称的参数
- 例如选择了 `grsai_api` 提供商，就选择标注了 `Grsai API` 的参数
- 如果选择了不支持的参数，API 可能会返回错误

---

## 自动保存

### 图片保存
当 `save_to_output` 设置为"启用"时，图片会保存到 `output/banana/` 目录，文件名格式：
```
banana_YYYYMMDD_HHMMSS_任务ID.png
```

### 文本保存
Gemini Vision 节点当 `save_response` 设置为"启用"时，响应文本会保存到 `output/banana/` 目录，文件名格式：
```
gemini_YYYYMMDD_HHMMSS.txt
```

---

## 日志输出

- **控制台日志**：显示所有详细信息，包括每次轮询
- **输出日志**：只显示关键信息，更简洁
- 日志带有 emoji 图标，方便快速识别状态

---

## 使用场景示例

### Banana Image Generation

#### 1. 纯文本生成
- 不连接任何图片输入
- 填写 prompt
- 运行

#### 2. 单图参考生成
- 连接一张图片到 image1
- 填写 prompt
- 运行

#### 3. 多图参考生成（最多 5 张）
- 连接多张图片到 image1-5
- 填写 prompt
- 运行

### Gemini Vision

#### 1. 图片描述
- 连接图片到 `image`
- prompt: "详细描述这张图片的内容"
- 获取文本描述

#### 2. 视频分析
- video_url: "https://example.com/video.mp4"
- prompt: "总结这个视频的主要内容"
- 获取视频分析

#### 3. 图文对比分析
- 连接图片到 `image`
- video_url: "https://example.com/video.mp4"
- prompt: "比较图片和视频中的场景差异"
- 获取对比分析

---

## 注意事项

- 首次使用请务必配置 `config.json` 并在 `api_keys` 中填写对应提供商的 API Key
- 切换 API 提供商时会自动使用对应的 key，无需手动修改
- 不同 API 提供商支持的模型和参数可能不同
- 高分辨率生成时间较长，建议增加 max_retries 和 timeout
- 错误时节点会直接报错并停止工作流
- 日志带有 emoji 图标，方便快速识别状态
- Gemini Vision 节点固定使用 `gemini-3.1-flash-lite-preview` 模型
- Gemini Vision 节点固定使用 `bltai_api` 的 API Key

---

## 依赖

- requests
- torch
- numpy
- PIL (Pillow)

这些依赖通常已经包含在 ComfyUI 环境中。

---

## 更新日志

### v1.1.0
- 新增 Gemini Vision API 节点
- 支持图片和视频分析
- 使用 bltai_api 的 API Key

### v1.0.0
- 初始版本
- 支持 Banana Image Generation API
- 支持多个第三方 API 提供商
