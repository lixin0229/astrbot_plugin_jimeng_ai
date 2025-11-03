import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from astrbot.api import (
    AstrBotMessage,
    CommandResult,
    Context,
    LLMToolCall,
    LLMToolResult,
    MessageChain,
    Plain,
    Image,
    logger,
    register,
)

from .utils.jimeng_api import generate_image_jimeng


class JiMengAIPlugin:
    """即梦AI绘图插件"""

    def __init__(self, context: Context):
        self.context = context
        self.config = context.config_helper.get_all()
        
        # 验证配置
        self._validate_config()
        
        # 注册命令和LLM工具
        register.command(
            "jimeng", 
            "即梦AI绘图", 
            "使用即梦AI生成图像\n用法: /jimeng <提示词> [参数]\n参数: --model <模型> --size <宽度>x<高度> --strength <精细度>",
            1
        )(self.jimeng_command)
        
        register.llm_tool(
            "jimeng_ai_image_generation",
            "即梦AI图像生成工具",
            {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "图像生成的提示词，描述要生成的图像内容"
                    },
                    "negative_prompt": {
                        "type": "string", 
                        "description": "反向提示词，描述不希望出现的内容",
                        "default": ""
                    },
                    "model": {
                        "type": "string",
                        "description": "使用的模型名称",
                        "default": "jimeng-3.0"
                    },
                    "width": {
                        "type": "integer",
                        "description": "图像宽度 (64-2048)",
                        "default": 1024
                    },
                    "height": {
                        "type": "integer", 
                        "description": "图像高度 (64-2048)",
                        "default": 1024
                    },
                    "sample_strength": {
                        "type": "number",
                        "description": "生成精细度 (0.0-1.0)",
                        "default": 0.5
                    }
                },
                "required": ["prompt"]
            }
        )(self.llm_image_generation)

    def _validate_config(self):
        """验证插件配置"""
        required_fields = ["api_base_url", "api_tokens"]
        for field in required_fields:
            if not self.config.get(field):
                logger.error(f"即梦AI插件配置缺失: {field}")
                raise ValueError(f"配置项 {field} 不能为空")
        
        # 处理token格式
        tokens = self.config.get("api_tokens", "")
        if isinstance(tokens, str):
            # 支持逗号分隔的多个token
            self.api_tokens = [token.strip() for token in tokens.split(",") if token.strip()]
        elif isinstance(tokens, list):
            self.api_tokens = tokens
        else:
            raise ValueError("api_tokens 必须是字符串或列表")
        
        if not self.api_tokens:
            raise ValueError("至少需要提供一个API token")
        
        logger.info(f"即梦AI插件已加载，配置了 {len(self.api_tokens)} 个token")

    async def jimeng_command(self, message: AstrBotMessage) -> CommandResult:
        """处理 /jimeng 命令"""
        try:
            # 解析命令参数
            args = self._parse_command_args(message.message)
            
            if not args.get("prompt"):
                return CommandResult().message("❌ 请提供图像生成提示词\n用法: /jimeng <提示词> [参数]")
            
            # 检查群组权限
            if not self._check_group_permission(message):
                return CommandResult().message("❌ 此群组未开启即梦AI绘图功能")
            
            # 生成图像
            result_msg = await self._generate_image_with_feedback(args, message)
            return CommandResult().message(result_msg)
            
        except Exception as e:
            logger.error(f"即梦AI命令处理失败: {e}")
            return CommandResult().message(f"❌ 处理失败: {str(e)}")

    async def llm_image_generation(self, tool_call: LLMToolCall) -> LLMToolResult:
        """LLM工具：图像生成"""
        try:
            args = tool_call.arguments
            
            # 生成图像
            image_url, image_path = await generate_image_jimeng(
                prompt=args["prompt"],
                api_tokens=self.api_tokens,
                api_base_url=self.config["api_base_url"],
                model=args.get("model", self.config.get("default_model", "jimeng-3.0")),
                negative_prompt=args.get("negative_prompt", ""),
                width=args.get("width", self.config.get("default_width", 1024)),
                height=args.get("height", self.config.get("default_height", 1024)),
                sample_strength=args.get("sample_strength", self.config.get("default_sample_strength", 0.5)),
                max_retry_attempts=self.config.get("max_retry_attempts", 3),
                timeout_seconds=self.config.get("timeout_seconds", 60),
            )
            
            if image_path:
                return LLMToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    content=f"✅ 图像生成成功！\n提示词: {args['prompt']}\n图像已保存到: {image_path}"
                )
            elif image_url:
                return LLMToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    content=f"✅ 图像生成成功！\n提示词: {args['prompt']}\n图像URL: {image_url}"
                )
            else:
                return LLMToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    content=f"❌ 图像生成失败，请稍后重试"
                )
                
        except Exception as e:
            logger.error(f"LLM图像生成工具失败: {e}")
            return LLMToolResult(
                tool_call_id=tool_call.tool_call_id,
                content=f"❌ 图像生成失败: {str(e)}"
            )

    def _parse_command_args(self, message_text: str) -> Dict:
        """解析命令参数"""
        # 移除命令前缀
        text = re.sub(r'^/jimeng\s*', '', message_text, flags=re.IGNORECASE).strip()
        
        args = {}
        
        # 解析参数
        # --model <模型名>
        model_match = re.search(r'--model\s+(\S+)', text)
        if model_match:
            args["model"] = model_match.group(1)
            text = re.sub(r'--model\s+\S+', '', text).strip()
        
        # --size <宽度>x<高度>
        size_match = re.search(r'--size\s+(\d+)x(\d+)', text)
        if size_match:
            args["width"] = int(size_match.group(1))
            args["height"] = int(size_match.group(2))
            text = re.sub(r'--size\s+\d+x\d+', '', text).strip()
        
        # --strength <精细度>
        strength_match = re.search(r'--strength\s+([\d.]+)', text)
        if strength_match:
            args["sample_strength"] = float(strength_match.group(1))
            text = re.sub(r'--strength\s+[\d.]+', '', text).strip()
        
        # --negative <反向提示词>
        negative_match = re.search(r'--negative\s+(.+?)(?=\s+--|$)', text)
        if negative_match:
            args["negative_prompt"] = negative_match.group(1).strip()
            text = re.sub(r'--negative\s+.+?(?=\s+--|$)', '', text).strip()
        
        # 剩余的文本作为主提示词
        if text:
            args["prompt"] = text
        
        return args

    def _check_group_permission(self, message: AstrBotMessage) -> bool:
        """检查群组权限"""
        if not self.config.get("enable_group_control", False):
            return True
        
        # 如果是私聊，总是允许
        if not hasattr(message, 'session_id') or not message.session_id:
            return True
        
        # 检查群组白名单
        allowed_groups = self.config.get("allowed_groups", [])
        if not allowed_groups:
            return True
        
        return str(message.session_id) in [str(g) for g in allowed_groups]

    async def _generate_image_with_feedback(self, args: Dict, message: AstrBotMessage) -> Union[MessageChain, str]:
        """生成图像并提供反馈"""
        # 发送开始生成的消息
        prompt = args["prompt"]
        model = args.get("model", self.config.get("default_model", "jimeng-3.0"))
        
        # 生成图像
        image_url, image_path = await generate_image_jimeng(
            prompt=prompt,
            api_tokens=self.api_tokens,
            api_base_url=self.config["api_base_url"],
            model=model,
            negative_prompt=args.get("negative_prompt", ""),
            width=args.get("width", self.config.get("default_width", 1024)),
            height=args.get("height", self.config.get("default_height", 1024)),
            sample_strength=args.get("sample_strength", self.config.get("default_sample_strength", 0.5)),
            max_retry_attempts=self.config.get("max_retry_attempts", 3),
            timeout_seconds=self.config.get("timeout_seconds", 60),
        )
        
        if image_path:
            # 构建消息链
            chain = MessageChain([
                Plain(f"✅ 即梦AI图像生成完成！\n"),
                Plain(f"📝 提示词: {prompt}\n"),
                Plain(f"🎨 模型: {model}\n"),
                Plain(f"📐 尺寸: {args.get('width', 1024)}x{args.get('height', 1024)}\n"),
                Image(path=image_path)
            ])
            return chain
        elif image_url:
            return f"✅ 即梦AI图像生成完成！\n📝 提示词: {prompt}\n🎨 模型: {model}\n🔗 图像URL: {image_url}"
        else:
            return f"❌ 图像生成失败，请检查配置或稍后重试\n📝 提示词: {prompt}"


def register_plugin(context: Context):
    """注册插件"""
    return JiMengAIPlugin(context)