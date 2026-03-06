# Banana Image Generation API - ComfyUI 节点

这是一个支持多个第三方 API 提供商的 Banana 图像生成 ComfyUI 节点。

## 功能特性

- 支持多个第三方 API 提供商（可扩展）
- 支持多张图片输入（最多 5 张，可选）
- 支持纯文本 prompt 生成
- 可配置图片尺寸和宽高比
- 海外和国内 Host 快速切换
- 自动轮询获取结果
- 可选自动保存到 output/banana 目录
- 详细日志输出（带 emoji 图标）

## 安装

1. 将此文件夹放置在 ComfyUI 的 `custom_nodes` 目录下
2. 复制 `config.json.example` 为 `config.json`
3. 编辑 `config.json` 填入你的 API Key
4. 重启 ComfyUI

## 配置文件

在 `Grsai_Api` 目录下创建 `config.json` 文件：

```json
{
  "api_keys": {
    "grsai_api": "your_grsai_api_key_here",
    "bltai_api": "your_bltai_api_key_here"
  }
}
```

配置说明：
- `api_keys`: 存储各个 API 提供商的密钥
  - 键名必须对应 `api/` 目录下的配置文件名（不含 .json 后缀）
  - 例如：`grsai_api` 对应 `api/grsai_api.json`
  - 切换提供商时会自动使用对应的 key

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
  - Host: 需要用户提供
  - 支持同步返回结果
  - 模型: gemini-3.1-flash-image-preview, nano-banana-2

### 添加新的 API 提供商

在 `api/` 目录下创建新的 JSON 配置文件，参考 `grsai_api.json` 的格式。重启 ComfyUI 后，新的提供商会自动加载。

## 使用方法

### 节点参数

- **api_provider**: API 提供商选择
- **host_type**: Host 类型（china/overseas/custom）
- **custom_host**: 自定义 Host 地址
- **api_key_override**: 临时覆盖 API Key（留空则使用配置文件中对应提供商的 key）
- **image1 ~ image5**: 5 个图片输入端口（可选）
- **image1_desc ~ image5_desc**: 图片描述（用于日志）
- **prompt**: 提示词（必填）
- **model**: 模型选择（格式：`模型名 (提供商1, 提供商2)`）
- **aspect_ratio**: 宽高比（格式：`比例 (提供商1, 提供商2)`）
- **image_size**: 输出图片尺寸（格式：`尺寸 (提供商1, 提供商2)`）
- **timeout**: 请求超时时间（秒）
- **max_retries**: 最大重试次数
- **poll_interval**: 轮询间隔（秒）
- **save_to_output**: 是否保存到 output/banana 目录

### 参数显示说明

为了方便识别哪些参数属于哪个提供商，节点会在参数后面标注支持的提供商：

**示例：**
- `nano-banana-fast (Grsai API)` - 只有 Grsai API 支持
- `1:1 (Grsai API, Custom API)` - 两个提供商都支持
- `auto (All)` - 所有提供商都支持

**使用建议：**
- 选择带有当前提供商名称的参数
- 例如选择了 `grsai_api` 提供商，就选择标注了 `Grsai API` 的参数
- 如果选择了不支持的参数，API 可能会返回错误

### 输出

- **result_image**: 生成的结果图片
- **log**: 详细的处理日志（带 emoji 图标）

### 自动保存

当 `save_to_output` 设置为"启用"时，图片会保存到 `output/banana/` 目录，文件名格式：
```
banana_YYYYMMDD_HHMMSS_任务ID.png
```

### 日志输出

- **控制台日志**：显示所有详细信息，包括每次轮询
- **输出日志**：只显示关键信息，更简洁

## 使用场景

### 1. 纯文本生成
- 不连接任何图片输入
- 填写 prompt
- 运行

### 2. 单图参考生成
- 连接一张图片到 image1
- 填写 prompt
- 运行

### 3. 多图参考生成（最多 5 张）
- 连接多张图片到 image1-5
- 填写 prompt
- 运行

## 注意事项

- 首次使用请务必配置 `config.json` 并在 `api_keys` 中填写对应提供商的 API Key
- 切换 API 提供商时会自动使用对应的 key，无需手动修改
- 可以在节点的 `api_key_override` 参数中临时覆盖 API Key
- 不同 API 提供商支持的模型和参数可能不同
- 高分辨率生成时间较长，建议增加 max_retries 和 timeout
- 错误时节点会直接报错并停止工作流
- 日志带有 emoji 图标，方便快速识别状态

## 依赖

- requests
- torch
- numpy
- PIL (Pillow)

这些依赖通常已经包含在 ComfyUI 环境中。
