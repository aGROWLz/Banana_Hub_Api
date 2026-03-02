# Grsai Nano Banana ComfyUI 节点

这是一个用于调用 Grsai Nano Banana API 的 ComfyUI 自定义节点。

## 功能特性

- 支持多张图片输入（最多 5 张，可选）
- 支持纯文本 prompt 生成
- 支持配置文件存储 API Key 和 Host
- 支持所有 Nano Banana 系列模型
- 可配置图片尺寸（1K/2K/4K）和宽高比
- 预设海外和国内 Host 快速切换
- 自动轮询获取结果
- 自动保存生成的图片到 output/banana 目录
- 输出结果图片和详细日志（带 emoji 图标）

## 安装

1. 将此文件夹放置在 ComfyUI 的 `custom_nodes` 目录下
2. 复制 `config.json.example` 为 `config.json`
3. 编辑 `config.json` 填入你的 API Key
4. 重启 ComfyUI

## 配置文件

在 `Grsai_Api` 目录下创建 `config.json` 文件：

```json
{
  "api_host": "https://grsai.dakka.com.cn",
  "api_key": "your_api_key_here",
  "host_overseas": "https://grsaiapi.com",
  "host_china": "https://grsai.dakka.com.cn"
}
```

配置说明：
- `api_host`: 默认使用的 API 主机地址（默认国内直连）
- `api_key`: 你的 API 密钥（必填）
- `host_overseas`: 海外 Host 地址
- `host_china`: 国内直连 Host 地址（默认）

## 使用方法

### 节点参数

- **image1 ~ image5**: 5 个独立的图片输入端口（全部可选）
- **prompt**: 提示词（必填）
- **model**: 模型选择
  - nano-banana-2
  - nano-banana-fast（默认，速度快）
  - nano-banana
  - nano-banana-pro
  - nano-banana-pro-vt
  - nano-banana-pro-cl
  - nano-banana-pro-vip（支持 1K/2K）
  - nano-banana-pro-4k-vip（支持 4K）
- **aspect_ratio**: 宽高比
  - auto（自动）
  - 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 5:4, 4:5, 21:9
- **image_size**: 输出图片尺寸
  - 1K（默认）
  - 2K（部分模型支持）
  - 4K（仅 nano-banana-pro-4k-vip 支持）
- **api_host_preset**: API Host 选择
  - 使用配置文件（默认）
  - 海外 Host
  - 国内 Host
  - 自定义
- **custom_host**: 自定义 Host 地址（当选择"自定义"时使用）
- **api_key_override**: API Key 覆盖（留空则使用配置文件中的 api_key）
- **timeout**: 请求超时时间（秒，默认 120）
- **max_retries**: 最大重试次数（默认 20）
- **poll_interval**: 轮询间隔（秒，默认 3）
- **save_to_output**: 保存到 output 目录
  - 启用（默认）：自动保存到 output/banana 目录
  - 禁用：不保存，只返回图片数据

### 输出

- **result_image**: 生成的结果图片
- **log**: 详细的处理日志（文本格式，带 emoji 图标）

### 自动保存

当 `save_to_output` 参数设置为"启用"时，生成的图片会自动保存到 ComfyUI 的 `output/banana` 目录下，文件名格式：
```
banana_YYYYMMDD_HHMMSS_任务ID.png
```

例如：`banana_20260228_143052_bc94c2c4.png`

如果不需要保存，可以将参数设置为"禁用"。

### 日志输出

节点提供两种日志：
- **控制台日志**：显示所有详细信息，包括每次轮询的状态（用于调试）
- **输出日志**：只显示关键信息，不包含重复的轮询记录（更简洁）

## 使用场景

### 1. 纯文本生成图片
- 不连接任何 image 输入
- 填写 prompt
- 选择模型和参数
- 运行

### 2. 单图参考生成
- 连接一张图片到 image1
- 填写 prompt 描述想要的效果
- 选择模型和参数
- 运行

### 3. 多图参考生成（最多 5 张）
- 连接多张图片到 image1, image2, image3, image4, image5
- 填写 prompt 描述想要的效果
- 选择模型和参数
- 运行

## 模型说明

根据 API 文档，不同模型支持的分辨率：
- nano-banana-2: 支持 1K, 2K, 4K
- nano-banana-pro: 支持 1K, 2K, 4K
- nano-banana-pro-vt: 支持 1K, 2K, 4K
- nano-banana-pro-cl: 支持 1K, 2K, 4K
- nano-banana-pro-vip: 仅支持 1K, 2K
- nano-banana-pro-4k-vip: 仅支持 4K

注意：分辨率越高，生成时间越长

## 注意事项

- 首次使用请务必配置 `config.json` 文件并填写 API Key
- 默认使用国内直连 Host（https://grsai.dakka.com.cn）
- API Key 存储在配置文件中，避免每次手动输入
- 可以在节点中临时覆盖 API Key（使用 api_key_override 参数）
- 根据网络情况选择合适的 Host（海外或国内）
- 高分辨率（2K/4K）生成时间较长，建议增加 max_retries 和 timeout
- 如果遇到 "error" 失败原因，可以尝试重新提交任务
- 错误时节点会直接报错并停止工作流，不会输出黑色图片
- 生成的图片会自动保存到 `output/banana` 目录
- 日志输出带有 emoji 图标，方便快速识别状态

## 依赖

- requests
- torch
- numpy
- PIL (Pillow)

这些依赖通常已经包含在 ComfyUI 环境中。
