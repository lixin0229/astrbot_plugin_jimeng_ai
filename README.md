# 即梦AI绘图插件

基于即梦AI接口的AstrBot图像生成插件，支持多token轮询和丰富的参数配置。

## 功能特性

- 🎨 支持即梦AI多种模型的图像生成
- 🔄 多API token轮询，提高可用性
- ⚙️ 丰富的参数配置（尺寸、精细度、反向提示词等）
- 🤖 LLM工具集成，支持AI助手调用
- 👥 群组权限控制
- 📱 支持命令行和对话式调用

## 安装配置

1. 将插件文件夹放置到AstrBot的plugins目录
2. 在插件配置中填写以下信息：

### 必需配置

- `api_base_url`: 即梦AI的API的逆向地址 (如: `http://xxxxx:8101`)
- `api_tokens`: API token列表，支持多个token用逗号分隔

### 可选配置

- `default_model`: 默认模型 (默认: `jimeng-3.0`)
- `default_width`: 默认图像宽度 (默认: `1024`)
- `default_height`: 默认图像高度 (默认: `1024`)
- `default_sample_strength`: 默认精细度 (默认: `0.5`)
- `max_retry_attempts`: 最大重试次数 (默认: `3`)
- `timeout_seconds`: 请求超时时间 (默认: `60`)
- `enable_group_control`: 是否启用群组控制 (默认: `false`)
- `allowed_groups`: 允许使用的群组ID列表

## 使用方法

### 命令行方式

```
/jimeng <提示词> [参数]
```

#### 参数说明

- `--model <模型名>`: 指定使用的模型
- `--size <宽度>x<高度>`: 指定图像尺寸 (如: `--size 512x768`)
- `--strength <精细度>`: 指定生成精细度 (0.0-1.0)
- `--negative <反向提示词>`: 指定不希望出现的内容

#### 使用示例

```bash
# 基础用法
/jimeng 一只可爱的小猫咪

# 指定参数
/jimeng 美丽的风景画 --model jimeng-3.0 --size 1024x768 --strength 0.8

# 使用反向提示词
/jimeng 科幻城市 --negative 模糊,低质量 --strength 0.7
```

### LLM工具调用

插件自动注册为LLM工具，AI助手可以直接调用进行图像生成：

```json
{
  "tool_name": "jimeng_ai_image_generation",
  "arguments": {
    "prompt": "一幅抽象艺术画",
    "negative_prompt": "模糊,低质量",
    "model": "jimeng-3.0",
    "width": 1024,
    "height": 1024,
    "sample_strength": 0.6
  }
}
```

## 配置示例

```json
{
  "api_base_url": "http://114.66.58.77:8101",
  "api_tokens": "token1,token2,token3",
  "default_model": "jimeng-3.0",
  "default_width": 1024,
  "default_height": 1024,
  "default_sample_strength": 0.5,
  "max_retry_attempts": 3,
  "timeout_seconds": 60,
  "enable_group_control": true,
  "allowed_groups": ["123456789", "987654321"]
}
```

## 支持的模型

- `jimeng-3.0`: 即梦AI 3.0模型
- 其他模型请根据API文档配置

## 注意事项

1. 确保API地址和token的有效性
2. 图像尺寸范围：64-2048像素
3. 精细度范围：0.0-1.0
4. 多token配置可提高服务可用性
5. 生成的图像会保存在插件的`images`目录下

## 故障排除

### 常见问题

1. **401认证错误**: 检查API token是否正确
2. **404接口错误**: 检查API地址和模型名称
3. **超时错误**: 增加timeout_seconds配置
4. **生成失败**: 检查提示词是否合规，尝试调整参数

### 日志查看

插件会输出详细的日志信息，可通过AstrBot日志查看具体错误原因。

## 更新日志

### v1.0.0
- 初始版本发布
- 支持基础图像生成功能
- 多token轮询机制
- LLM工具集成