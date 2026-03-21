#!/usr/bin/env python3
"""
抖音无水印视频下载并提取文本的 MCP 服务器

该服务器提供以下功能：
1. 解析抖音分享链接获取无水印视频链接
2. 下载视频并提取音频
3. 从音频中提取文本内容
4. 自动清理中间文件

修改记录:
- 2026-03-21: 将语音识别从阿里云百炼(dashscope)迁移到硅基流动(SiliconFlow)
             使用 FunAudioLLM/SenseVoiceSmall 模型，与 CLI 保持一致
"""

import os
import re
import json
import requests
import tempfile
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import ffmpeg
from tqdm.asyncio import tqdm

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Context


# 创建 MCP 服务器实例
mcp = FastMCP("Douyin MCP Server", 
              dependencies=["requests", "ffmpeg-python", "tqdm"])

# 请求头，模拟移动端访问
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) EdgiOS/121.0.2277.107 Version/17.0 Mobile/15E148 Safari/604.1'
}

# 硅基流动 API 配置
DEFAULT_API_BASE_URL = "https://api.siliconflow.cn/v1/audio/transcriptions"
DEFAULT_MODEL = "FunAudioLLM/SenseVoiceSmall"


class DouyinProcessor:
    """抖音视频处理器"""
    
    def __init__(self, api_key: str, api_base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.api_base_url = api_base_url or DEFAULT_API_BASE_URL
        self.model = model or DEFAULT_MODEL
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def __del__(self):
        """清理临时目录"""
        import shutil
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def parse_share_url(self, share_text: str) -> dict:
        """从分享文本中提取无水印视频链接"""
        # 提取分享链接
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', share_text)
        if not urls:
            raise ValueError("未找到有效的分享链接")
        
        share_url = urls[0]
        share_response = requests.get(share_url, headers=HEADERS)
        video_id = share_response.url.split("?")[0].strip("/").split("/")[-1]
        share_url = f'https://www.iesdouyin.com/share/video/{video_id}'
        
        # 获取视频页面内容
        response = requests.get(share_url, headers=HEADERS)
        response.raise_for_status()
        
        pattern = re.compile(
            pattern=r"window\._ROUTER_DATA\s*=\s*(.*?)</script>",
            flags=re.DOTALL,
        )
        find_res = pattern.search(response.text)

        if not find_res or not find_res.group(1):
            raise ValueError("从HTML中解析视频信息失败")

        # 解析JSON数据
        json_data = json.loads(find_res.group(1).strip())
        VIDEO_ID_PAGE_KEY = "video_(id)/page"
        NOTE_ID_PAGE_KEY = "note_(id)/page"
        
        if VIDEO_ID_PAGE_KEY in json_data["loaderData"]:
            original_video_info = json_data["loaderData"][VIDEO_ID_PAGE_KEY]["videoInfoRes"]
        elif NOTE_ID_PAGE_KEY in json_data["loaderData"]:
            original_video_info = json_data["loaderData"][NOTE_ID_PAGE_KEY]["videoInfoRes"]
        else:
            raise Exception("无法从JSON中解析视频或图集信息")

        data = original_video_info["item_list"][0]

        # 获取视频信息
        video_url = data["video"]["play_addr"]["url_list"][0].replace("playwm", "play")
        desc = data.get("desc", "").strip() or f"douyin_{video_id}"
        
        # 替换文件名中的非法字符
        desc = re.sub(r'[\\/:*?"<>|]', '_', desc)
        
        return {
            "url": video_url,
            "title": desc,
            "video_id": video_id
        }
    
    async def download_video(self, video_info: dict, ctx: Context) -> Path:
        """异步下载视频到临时目录"""
        filename = f"{video_info['video_id']}.mp4"
        filepath = self.temp_dir / filename
        
        ctx.info(f"正在下载视频: {video_info['title']}")
        
        response = requests.get(video_info['url'], headers=HEADERS, stream=True)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 异步下载文件，显示进度
        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        await ctx.report_progress(downloaded, total_size)
        
        ctx.info(f"视频下载完成: {filepath}")
        return filepath
    
    def extract_audio(self, video_path: Path) -> Path:
        """从视频文件中提取音频"""
        audio_path = video_path.with_suffix('.mp3')
        
        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(str(audio_path), acodec='libmp3lame', q=0)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
            return audio_path
        except Exception as e:
            raise Exception(f"提取音频时出错: {str(e)}")
    
    def extract_text_from_audio(self, audio_path: Path) -> str:
        """从音频文件中提取文字（使用硅基流动API）"""
        try:
            # 检查文件大小，如果超过50MB需要分段处理
            file_size = audio_path.stat().st_size
            max_size = 50 * 1024 * 1024  # 50MB
            
            if file_size > max_size:
                return self._extract_text_from_large_audio(audio_path)
            
            return self._extract_text_from_small_audio(audio_path)
            
        except Exception as e:
            raise Exception(f"提取文字时出错: {str(e)}")
    
    def _extract_text_from_large_audio(self, audio_path: Path) -> str:
        """处理大文件音频（超过50MB）"""
        # 使用ffmpeg分割音频为9分钟一段
        segment_duration = 540  # 9分钟
        segment_pattern = self.temp_dir / "segment_%03d.mp3"
        
        try:
            subprocess.run([
                'ffmpeg', '-i', str(audio_path), 
                '-f', 'segment', '-segment_time', str(segment_duration),
                '-c', 'copy', str(segment_pattern)
            ], check=True, capture_output=True)
            
            # 获取所有分段文件
            segments = sorted(self.temp_dir.glob("segment_*.mp3"))
            
            # 逐段识别
            all_texts = []
            for segment in segments:
                text = self._extract_text_from_small_audio(segment)
                all_texts.append(text)
                segment.unlink()  # 删除已处理的分段
            
            return "\n".join(all_texts)
            
        except Exception as e:
            raise Exception(f"处理大文件音频时出错: {str(e)}")
    
    def _extract_text_from_small_audio(self, audio_path: Path) -> str:
        """处理小文件音频（小于50MB）"""
        with open(audio_path, 'rb') as audio_file:
            files = {
                'file': (audio_path.name, audio_file, 'audio/mpeg')
            }
            data = {
                'model': self.model
            }
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            
            response = requests.post(
                self.api_base_url,
                headers=headers,
                data=data,
                files=files
            )
            response.raise_for_status()
            result = response.json()
            
            # 提取文本内容
            if 'text' in result:
                return result['text']
            else:
                return "未识别到文本内容"
    
    def cleanup_files(self, *file_paths: Path):
        """清理指定的文件"""
        for file_path in file_paths:
            if file_path.exists():
                file_path.unlink()


@mcp.tool()
def get_douyin_download_link(share_link: str) -> str:
    """
    获取抖音视频的无水印下载链接
    
    参数:
    - share_link: 抖音分享链接或包含链接的文本
    
    返回:
    - 包含下载链接和视频信息的JSON字符串
    """
    try:
        processor = DouyinProcessor("")  # 获取下载链接不需要API密钥
        video_info = processor.parse_share_url(share_link)
        
        return json.dumps({
            "status": "success",
            "video_id": video_info["video_id"],
            "title": video_info["title"],
            "download_url": video_info["url"],
            "description": f"视频标题: {video_info['title']}",
            "usage_tip": "可以直接使用此链接下载无水印视频"
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": f"获取下载链接失败: {str(e)}"
        }, ensure_ascii=False, indent=2)


@mcp.tool()
async def extract_douyin_text(
    share_link: str,
    model: Optional[str] = None,
    ctx: Context = None
) -> str:
    """
    从抖音分享链接提取视频中的文本内容
    
    参数:
    - share_link: 抖音分享链接或包含链接的文本
    - model: 语音识别模型（可选，默认使用FunAudioLLM/SenseVoiceSmall）
    
    返回:
    - 提取的文本内容
    
    注意: 需要设置环境变量 API_KEY (硅基流动API密钥)
    """
    try:
        # 从环境变量获取API密钥
        api_key = os.getenv('API_KEY')
        if not api_key:
            raise ValueError("未设置环境变量 API_KEY，请在配置中添加硅基流动API密钥")
        
        processor = DouyinProcessor(api_key, model=model)
        
        # 解析视频链接
        ctx.info("正在解析抖音分享链接...")
        video_info = processor.parse_share_url(share_link)
        
        # 下载视频
        ctx.info("正在下载视频...")
        video_path = await processor.download_video(video_info, ctx)
        
        # 提取音频
        ctx.info("正在提取音频...")
        audio_path = processor.extract_audio(video_path)
        
        # 提取文本
        ctx.info("正在提取文本...")
        text_content = processor.extract_text_from_audio(audio_path)
        
        # 清理临时文件
        processor.cleanup_files(video_path, audio_path)
        
        ctx.info("文本提取完成!")
        return text_content
        
    except Exception as e:
        ctx.error(f"处理过程中出现错误: {str(e)}")
        raise Exception(f"提取抖音视频文本失败: {str(e)}")


@mcp.tool()
def parse_douyin_video_info(share_link: str) -> str:
    """
    解析抖音分享链接，获取视频基本信息
    
    参数:
    - share_link: 抖音分享链接或包含链接的文本
    
    返回:
    - 视频信息（JSON格式字符串）
    """
    try:
        processor = DouyinProcessor("")  # 不需要API密钥来解析链接
        video_info = processor.parse_share_url(share_link)
        
        return json.dumps({
            "video_id": video_info["video_id"],
            "title": video_info["title"],
            "download_url": video_info["url"],
            "status": "success"
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, ensure_ascii=False, indent=2)


@mcp.resource("douyin://video/{video_id}")
def get_video_info(video_id: str) -> str:
    """
    获取指定视频ID的详细信息
    
    参数:
    - video_id: 抖音视频ID
    
    返回:
    - 视频详细信息
    """
    share_url = f"https://www.iesdouyin.com/share/video/{video_id}"
    try:
        processor = DouyinProcessor("")
        video_info = processor.parse_share_url(share_url)
        return json.dumps(video_info, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"获取视频信息失败: {str(e)}"


@mcp.prompt()
def douyin_text_extraction_guide() -> str:
    """抖音视频文本提取使用指南"""
    return """
# 抖音视频文本提取使用指南

## 功能说明
这个MCP服务器可以从抖音分享链接中提取视频的文本内容，以及获取无水印下载链接。

## 环境变量配置
请确保设置了以下环境变量：
- `API_KEY`: 硅基流动API密钥 (https://cloud.siliconflow.cn)

## 使用步骤
1. 复制抖音视频的分享链接
2. 在Claude Desktop配置中设置环境变量 API_KEY
3. 使用相应的工具进行操作

## 工具说明
- `extract_douyin_text`: 完整的文本提取流程（需要API密钥）
- `get_douyin_download_link`: 获取无水印视频下载链接（无需API密钥）
- `parse_douyin_video_info`: 仅解析视频基本信息
- `douyin://video/{video_id}`: 获取指定视频的详细信息

## Claude Desktop 配置示例
```json
{
  "mcpServers": {
    "douyin-mcp": {
      "command": "uvx",
      "args": ["douyin-mcp-server"],
      "env": {
        "API_KEY": "your-siliconflow-api-key-here"
      }
    }
  }
}
```

## 注意事项
- 需要提供有效的硅基流动API密钥（通过环境变量）
- 使用硅基流动的FunAudioLLM/SenseVoiceSmall模型进行语音识别
- 支持大部分抖音视频格式
- 获取下载链接无需API密钥
- 大文件（超过50MB）会自动分段处理
"""


def main():
    """启动MCP服务器"""
    mcp.run()


if __name__ == "__main__":
    main()
