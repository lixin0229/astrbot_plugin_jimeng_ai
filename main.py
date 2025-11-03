import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.all import *
from astrbot.core.message.components import Reply, Image
from typing import Optional, List, Tuple, Dict, Union
import asyncio
import base64
import json
import uuid
import re
from datetime import datetime
from pathlib import Path
import httpx


class _TokenState:
    """Token轮询状态管理"""
    def __init__(self):
        self.token_index = 0
        self._lock = asyncio.Lock()

    async def get_next_token(self, tokens: List[str]) -> str:
        """获取下一个可用的token"""
        async with self._lock:
            if not tokens:
                raise ValueError("Token列表为空")
            token = tokens[self.token_index % len(tokens)]
            return token

    async def rotate(self, tokens: List[str]):
        """轮换到下一个token"""
        async with self._lock:
            if tokens:
                self.token_index = (self.token_index + 1) % len(tokens)


_token_state = _TokenState()


async def _save_image_bytes(content: bytes, suffix: str = "png") -> str:
    """保存图像字节数据到文件"""
    plugin_root = Path(__file__).parent
    images_dir = plugin_root / "images"
    images_dir.mkdir(exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    file_path = images_dir / f"jimeng_image_{ts}_{uid}.{suffix}"
    
    file_path.write_bytes(content)
    return str(file_path)


async def _decode_and_save_base64(data_b64: str) -> str:
    """解码base64图像数据并保存"""
    # 处理data URL格式
    if data_b64.startswith("data:"):
        try:
            header, b64_data = data_b64.split(",", 1)
            data_b64 = b64_data
        except Exception:
            pass
    
    try:
        image_bytes = base64.b64decode(data_b64)
        return await _save_image_bytes(image_bytes)
    except Exception as e:
        logger.error(f"解码base64图像失败: {e}")
        raise


async def _download_image_from_url(image_url: str, timeout_seconds: int = 30) -> str:
    """从URL下载图片并保存到本地"""
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            
            # 从URL或Content-Type推断文件扩展名
            content_type = response.headers.get('content-type', '').lower()
            if 'jpeg' in content_type or 'jpg' in content_type:
                suffix = 'jpg'
            elif 'png' in content_type:
                suffix = 'png'
            elif 'webp' in content_type:
                suffix = 'webp'
            elif 'gif' in content_type:
                suffix = 'gif'
            else:
                # 从URL推断
                if image_url.lower().endswith(('.jpg', '.jpeg')):
                    suffix = 'jpg'
                elif image_url.lower().endswith('.png'):
                    suffix = 'png'
                elif image_url.lower().endswith('.webp'):
                    suffix = 'webp'
                elif image_url.lower().endswith('.gif'):
                    suffix = 'gif'
                else:
                    suffix = 'png'  # 默认
            
            image_path = await _save_image_bytes(response.content, suffix)
            logger.info(f"成功从URL下载图片: {image_url} -> {image_path}")
            return image_path
            
    except Exception as e:
        logger.error(f"从URL下载图片失败 {image_url}: {e}")
        raise


async def generate_image_jimeng(
    prompt: str,
    api_tokens: List[str],
    api_base_url: str,
    model: str = "jimeng-3.0",
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    sample_strength: float = 0.5,
    max_retry_attempts: int = 3,
    timeout_seconds: int = 60,
) -> List[str]:
    """
    使用即梦AI生成图像
    
    Args:
        prompt: 提示词
        api_tokens: API token列表
        api_base_url: API基础地址
        model: 模型名称
        negative_prompt: 反向提示词
        width: 图像宽度
        height: 图像高度
        sample_strength: 精细度 (0.0-1.0)
        max_retry_attempts: 最大重试次数
        timeout_seconds: 超时时间
    
    Returns:
        List[str]: 本地图片路径列表
    """
    if isinstance(api_tokens, str):
        api_tokens = [api_tokens]

    if not api_tokens:
        logger.error("未提供API token")
        return []

    # 验证参数
    sample_strength = max(0.0, min(1.0, sample_strength))
    width = max(64, min(2048, width))
    height = max(64, min(2048, height))

    # 尝试每个token
    for token_attempt in range(len(api_tokens)):
        current_token = await _token_state.get_next_token(api_tokens)

        for attempt in range(max_retry_attempts):
            if attempt > 0:
                # 指数退避
                await asyncio.sleep(min(2 ** attempt, 10))

            try:
                url = f"{api_base_url.rstrip('/')}/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {current_token}"
                }

                # 使用OpenAI格式的messages，同时包含即梦AI的参数
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "prompt": prompt,
                    "negativePrompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "sample_strength": sample_strength
                }

                logger.info(f"即梦AI请求: {model}, 尺寸: {width}x{height}, 精细度: {sample_strength}")
                logger.debug(f"请求URL: {url}")
                logger.debug(f"提示词: {prompt[:100]}...")

                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    response = await client.post(url, headers=headers, json=payload)

                    if response.status_code == 200:
                        try:
                            data = response.json()
                            
                            # 检查响应格式
                            if "error" in data:
                                logger.error(f"即梦AI API错误: {data['error']}")
                                continue
                            
                            # 收集所有图片数据
                            image_paths = []
                            
                            # 格式1: OpenAI格式的choices
                            if "choices" in data and data["choices"]:
                                for choice in data["choices"]:
                                    if "message" in choice and "content" in choice["message"]:
                                        content = choice["message"]["content"]
                                        # 检查是否包含图像URL
                                        if "![image_" in content and "https://" in content:
                                            # 提取所有URL
                                            import re
                                            urls = re.findall(r'https://[^\s\)]+', content)
                                            for image_url in urls:
                                                try:
                                                    image_path = await _download_image_from_url(image_url, timeout_seconds)
                                                    image_paths.append(image_path)
                                                except Exception as e:
                                                    logger.error(f"下载图片失败: {e}")
                                        elif isinstance(content, str) and len(content) > 100:
                                            # 可能是base64数据
                                            try:
                                                image_path = await _decode_and_save_base64(content)
                                                image_paths.append(image_path)
                                            except Exception as e:
                                                logger.error(f"处理base64数据失败: {e}")
                            
                            # 格式2: 直接返回数据数组
                            elif "data" in data:
                                data_list = data["data"] if isinstance(data["data"], list) else [data["data"]]
                                for item in data_list:
                                    if isinstance(item, str):
                                        if item.startswith("http"):
                                            # URL格式
                                            try:
                                                image_path = await _download_image_from_url(item, timeout_seconds)
                                                image_paths.append(image_path)
                                            except Exception as e:
                                                logger.error(f"下载图片失败: {e}")
                                        else:
                                            # base64格式
                                            try:
                                                image_path = await _decode_and_save_base64(item)
                                                image_paths.append(image_path)
                                            except Exception as e:
                                                logger.error(f"处理base64数据失败: {e}")
                                    elif isinstance(item, dict):
                                        # 嵌套对象格式
                                        if "url" in item:
                                            try:
                                                image_path = await _download_image_from_url(item["url"], timeout_seconds)
                                                image_paths.append(image_path)
                                            except Exception as e:
                                                logger.error(f"下载图片失败: {e}")
                                        elif "data" in item:
                                            try:
                                                image_path = await _decode_and_save_base64(item["data"])
                                                image_paths.append(image_path)
                                            except Exception as e:
                                                logger.error(f"处理base64数据失败: {e}")
                            
                            # 格式3: 直接在根级别的images数组
                            elif "images" in data:
                                images_list = data["images"] if isinstance(data["images"], list) else [data["images"]]
                                for item in images_list:
                                    if isinstance(item, str):
                                        if item.startswith("http"):
                                            try:
                                                image_path = await _download_image_from_url(item, timeout_seconds)
                                                image_paths.append(image_path)
                                            except Exception as e:
                                                logger.error(f"下载图片失败: {e}")
                                        else:
                                            try:
                                                image_path = await _decode_and_save_base64(item)
                                                image_paths.append(image_path)
                                            except Exception as e:
                                                logger.error(f"处理base64数据失败: {e}")
                                    elif isinstance(item, dict) and "url" in item:
                                        try:
                                            image_path = await _download_image_from_url(item["url"], timeout_seconds)
                                            image_paths.append(image_path)
                                        except Exception as e:
                                            logger.error(f"下载图片失败: {e}")
                            
                            # 格式4: 单个URL或base64
                            elif "url" in data:
                                try:
                                    image_path = await _download_image_from_url(data["url"], timeout_seconds)
                                    image_paths.append(image_path)
                                except Exception as e:
                                    logger.error(f"下载图片失败: {e}")
                            elif "image" in data:
                                if isinstance(data["image"], str):
                                    if data["image"].startswith("http"):
                                        try:
                                            image_path = await _download_image_from_url(data["image"], timeout_seconds)
                                            image_paths.append(image_path)
                                        except Exception as e:
                                            logger.error(f"下载图片失败: {e}")
                                    else:
                                        try:
                                            image_path = await _decode_and_save_base64(data["image"])
                                            image_paths.append(image_path)
                                        except Exception as e:
                                            logger.error(f"处理base64数据失败: {e}")
                            
                            if image_paths:
                                logger.info(f"✅ 即梦AI成功生成 {len(image_paths)} 张图像")
                                return image_paths
                            else:
                                logger.warning(f"未找到图像数据，响应结构: {json.dumps(data, indent=2)[:500]}...")
                                continue

                        except json.JSONDecodeError as e:
                            logger.error(f"解析JSON响应失败: {e}")
                            logger.debug(f"响应内容: {response.text[:200]}...")
                            continue

                    elif response.status_code == 401:
                        logger.warning(f"Token认证失败，尝试下一个token")
                        break  # 跳出重试循环，尝试下一个token
                    
                    elif response.status_code == 429:
                        logger.warning(f"请求频率限制，等待后重试")
                        await asyncio.sleep(5)
                        continue
                    
                    else:
                        logger.error(f"即梦AI API请求失败: {response.status_code}")
                        logger.debug(f"响应内容: {response.text[:200]}...")
                        continue

            except httpx.TimeoutException:
                logger.warning(f"请求超时，重试中... (尝试 {attempt + 1}/{max_retry_attempts})")
                continue
            except Exception as e:
                logger.error(f"请求异常: {e}")
                continue

        # 当前token失败，轮换到下一个
        await _token_state.rotate(api_tokens)

    logger.error("所有token都失败了")
    return []


@register("jimeng-ai", "lixin0229", "基于即梦AI接口的图像生成插件，支持多token轮询和丰富的参数配置", "1.0.0")
class JiMengAIPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config or context.config_helper.get_all()
        
        # 验证配置
        self._validate_config()

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

    @filter.command("jimeng")
    async def jimeng_command(self, event: AstrMessageEvent):
        """处理 /jimeng 命令"""
        try:
            # 解析命令参数
            args = self._parse_command_args(event.message_str)
            
            if not args.get("prompt"):
                yield event.plain_result("❌ 请提供图像生成提示词\n用法: /jimeng <提示词> [参数]")
                return
            
            # 检查群组权限
            if not self._check_group_permission(event):
                yield event.plain_result("❌ 此群组未开启即梦AI绘图功能")
                return
            
            # 生成图像
            result_msg = await self._generate_image_with_feedback(args, event)
            yield result_msg
            
        except Exception as e:
            logger.error(f"即梦AI命令处理失败: {e}")
            yield event.plain_result(f"❌ 处理失败: {str(e)}")

    @filter.llm_tool(name="jimeng_ai_image_generation")
    async def llm_image_generation(self, event: AstrMessageEvent, prompt: str):
        """
        LLM工具：使用即梦AI生成图像
        
        Args:
            prompt(string): 图像生成提示词，描述想要生成的图像内容
        """
        try:
            # 发送状态消息
            await event.send(event.plain_result("🎨 正在使用即梦AI为您生成图像，请稍候..."))
            
            # 生成图像，使用默认配置
            image_paths = await generate_image_jimeng(
                prompt=prompt,
                api_tokens=self.api_tokens,
                api_base_url=self.config["api_base_url"],
                model=self.config.get("default_model", "jimeng-3.0"),
                negative_prompt=self.config.get("default_negative_prompt", ""),
                width=self.config.get("default_width", 1024),
                height=self.config.get("default_height", 1024),
                sample_strength=self.config.get("default_sample_strength", 0.5),
                max_retry_attempts=self.config.get("max_retry_attempts", 3),
                timeout_seconds=self.config.get("timeout_seconds", 60),
            )
            
            if image_paths:
                # 构建包含图片的消息链
                message_components = [Plain(f"✅ 成功生成 {len(image_paths)} 张图像！\n提示词: {prompt}\n")]
                
                for i, image_path in enumerate(image_paths):
                    message_components.append(Image(path=image_path))
                    if i < len(image_paths) - 1:  # 不是最后一张图片时添加换行
                        message_components.append(Plain("\n"))
                
                result_message = MessageChain(message_components)
                await event.send(result_message)
            else:
                await event.send(event.plain_result(f"❌ 图像生成失败，请稍后重试"))
                
        except Exception as e:
            logger.error(f"LLM图像生成工具失败: {e}")
            await event.send(event.plain_result(f"❌ 图像生成失败: {str(e)}"))

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

    def _check_group_permission(self, event: AstrMessageEvent) -> bool:
        """检查群组权限"""
        if not self.config.get("enable_group_control", False):
            return True
        
        # 如果是私聊，总是允许
        if event.is_private_chat():
            return True
        
        # 检查群组白名单
        allowed_groups = self.config.get("allowed_groups", [])
        if not allowed_groups:
            return True
        
        group_id = event.get_group_id()
        return str(group_id) in [str(g) for g in allowed_groups] if group_id else False

    async def _generate_image_with_feedback(self, args: Dict, event: AstrMessageEvent) -> Union[MessageChain, str]:
        """生成图像并提供反馈"""
        # 发送开始生成的消息
        prompt = args["prompt"]
        model = args.get("model", self.config.get("default_model", "jimeng-3.0"))
        
        # 生成图像
        image_paths = await generate_image_jimeng(
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
        
        if image_paths:
            if len(image_paths) == 1:
                # 单张图片
                chain = MessageChain([
                    Plain(f"✅ 即梦AI图像生成完成！\n"),
                    Plain(f"📝 提示词: {prompt}\n"),
                    Plain(f"🎨 模型: {model}\n"),
                    Plain(f"📐 尺寸: {args.get('width', 1024)}x{args.get('height', 1024)}\n"),
                    Image(path=image_paths[0])
                ])
                return chain
            else:
                # 多张图片
                chain_items = [
                    Plain(f"✅ 即梦AI成功生成 {len(image_paths)} 张图像！\n"),
                    Plain(f"📝 提示词: {prompt}\n"),
                    Plain(f"🎨 模型: {model}\n"),
                    Plain(f"📐 尺寸: {args.get('width', 1024)}x{args.get('height', 1024)}\n\n")
                ]
                
                # 添加所有图片
                for i, image_path in enumerate(image_paths):
                    chain_items.append(Plain(f"图片 {i+1}:\n"))
                    chain_items.append(Image(path=image_path))
                    if i < len(image_paths) - 1:  # 不是最后一张图片
                        chain_items.append(Plain("\n"))
                
                return MessageChain(chain_items)
        else:
            return f"❌ 图像生成失败，请检查配置或稍后重试\n📝 提示词: {prompt}"

    async def terminate(self):
        """插件卸载时调用"""
        logger.info("即梦AI插件已卸载")
