"""Unsplash图片服务"""

import requests
from typing import List, Optional
from ..config import get_settings


def _sanitize_query(text: str, max_length: int = 100) -> str:
    """清洗搜索关键词：修复截断的UTF-8、限制长度、去除无效字符"""
    # 修复截断的UTF-8字节序列（用 errors='replace' 替换非法字节为 U+FFFD）
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    # 移除替换字符
    text = text.replace("�", "")
    # 移除 Latin-1 控制字符和孤立的 Latin-1 非 ASCII 字符
    # （UTF-8 多字节字符被截断后会变成 U+0080-U+00FF 范围的单字节）
    clean = []
    for ch in text:
        cp = ord(ch)
        if cp < 0x20 and cp not in (0x09, 0x0a, 0x0d):
            continue  # 控制字符（跳过）
        if 0x80 <= cp <= 0xFF and cp not in (0xa9, 0xae, 0xb0):
            continue  # Latin-1 非ASCII，仅保留 © ® °
        clean.append(ch)
    text = "".join(clean)
    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length]
    return text


class UnsplashService:
    """Unsplash图片服务类"""

    def __init__(self):
        """初始化服务"""
        settings = get_settings()
        self.access_key = settings.unsplash_access_key
        self.base_url = "https://api.unsplash.com"

    def search_photos(self, query: str, per_page: int = 5) -> List[dict]:
        """
        搜索图片

        Args:
            query: 搜索关键词
            per_page: 每页数量

        Returns:
            图片列表
        """
        query = _sanitize_query(query)
        if not query:
            return []

        try:
            url = f"{self.base_url}/search/photos"
            params = {
                "query": query,
                "per_page": per_page,
                "client_id": self.access_key,
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            # 提取图片URL
            photos = []
            for photo in results:
                photos.append({
                    "id": photo.get("id"),
                    "url": photo.get("urls", {}).get("regular"),
                    "thumb": photo.get("urls", {}).get("thumb"),
                    "description": photo.get("description") or photo.get("alt_description"),
                    "photographer": photo.get("user", {}).get("name"),
                })

            return photos

        except requests.HTTPError as e:
            status = e.response.status_code if hasattr(e, 'response') else 'unknown'
            print(f"[WARN] Unsplash HTTP {status}: {query}")
            return []
        except Exception as e:
            print(f"[WARN] Unsplash request failed: {query[:50]}")
            return []
    
    def get_photo_url(self, query: str) -> Optional[str]:
        """
        获取单张图片URL

        Args:
            query: 搜索关键词

        Returns:
            图片URL
        """
        photos = self.search_photos(query, per_page=1)
        if photos:
            return photos[0].get("url")
        return None


# 全局服务实例
_unsplash_service = None


def get_unsplash_service() -> UnsplashService:
    """获取Unsplash服务实例(单例模式)"""
    global _unsplash_service
    
    if _unsplash_service is None:
        _unsplash_service = UnsplashService()
    
    return _unsplash_service

