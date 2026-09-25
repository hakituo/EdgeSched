import json
import logging
import time
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class PersistentStateTracker:
    """
    Persistent State Tracking System
    
    Tracks the state of user actions/events (e.g., "ate dinner", "took medicine")
    to prevent the AI from making redundant suggestions.
    
    Features:
    - State storage with status (active, completed, ignored)
    - Time-to-Live (TTL) support for automatic expiration
    - Daily reset capability
    """
    
    def __init__(self, storage_dir: str, user_id: str = "default"):
        self.storage_dir = Path(storage_dir)
        self.user_id = user_id
        self.state_file = self.storage_dir / f"persistent_states_{user_id}.json"
        self.states: Dict[str, Dict[str, Any]] = {}
        self._load_states()
        
    def _load_states(self):
        """Load states from JSON file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    self.states = json.load(f)
                # Cleanup expired states on load
                self.cleanup_expired_states()
            except Exception as e:
                logger.error(f"Failed to load persistent states: {e}")
                self.states = {}
        else:
            self.states = {}
            
    def _save_states(self):
        """Save states to JSON file"""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.states, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save persistent states: {e}")

    def cleanup_expired_states(self):
        """Remove states that have exceeded their TTL"""
        current_time = time.time()
        expired_keys = []
        
        for key, state in self.states.items():
            expiry = state.get("expiry")
            if expiry and current_time > expiry:
                expired_keys.append(key)
                
        if expired_keys:
            for key in expired_keys:
                del self.states[key]
            self._save_states()
            logger.info(f"Cleaned up {len(expired_keys)} expired states for user {self.user_id}")

    def add_state(self, content: str, status: str = "completed", ttl_hours: int = 24, metadata: Dict = None):
        """
        Add or update a state.
        
        Args:
            content: Description of the state (e.g., "ate dinner")
            status: Status of the state (active, completed)
            ttl_hours: How long this state should persist (default 24h for daily tasks)
            metadata: Additional info
        """
        # 归一化内容：去除空格、转小写，并简单处理一些语气词
        normalized_content = content.strip().lower()
        # 移除末尾的语气词
        for suffix in ["了", "了哦", "啦", "哈", "呀", "呢"]:
            if normalized_content.endswith(suffix) and len(normalized_content) > len(suffix):
                normalized_content = normalized_content[:-len(suffix)]
                break

        # 使用 MD5 哈希作为 key，确保相同内容不会重复记录
        import hashlib
        key = hashlib.md5(normalized_content.encode('utf-8')).hexdigest()
        
        current_time = time.time()
        expiry = current_time + (ttl_hours * 3600) if ttl_hours else None
        
        self.states[key] = {
            "content": normalized_content, # 存储归一化后的核心动作
            "display_content": content,    # 保留原始描述用于展示
            "status": status,
            "created_at": current_time,
            "updated_at": current_time,
            "expiry": expiry,
            "metadata": metadata or {}
        }
        self._save_states()
        logger.info(f"Added persistent state: {normalized_content} ({status})")
        return key

    def extract_from_text(self, text: str) -> List[str]:
        """
        从文本中提取可能的状态更新。
        采用关键词触发 + 模式匹配的方案。
        """
        if not text:
            return []
            
        found_states = []
        
        # 1. 强触发词模式：识别“已经...了”、“吃过了”等
        patterns = [
            r"(?:已经|早就|确实)\s*(.+?)\s*(?:了|啦|完毕|完成|齐了)",
            r"(.+?)\s*(?:完了|好了|齐了|完毕|完成)",
            r"(?:做了|吃了|喝了|买了|搞定|完成)\s*(.+)",
        ]
        
        import re
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                state_content = match.group(1).strip()
                # 过滤太短或太长的无效提取
                if 1 < len(state_content) < 20:
                    found_states.append(state_content)
                    
        # 2. 补充：特定的动宾短语识别
        common_actions = ["吃饭", "吃过饭", "吃完饭", "洗澡", "睡觉", "吃药", "服药", "打卡"]
        for action in common_actions:
            if action in text and action not in found_states:
                # 检查是否有否定词
                if not any(neg in text for neg in ["没", "不", "还没"]):
                    found_states.append(action)
                    
        return list(set(found_states))

    def auto_update_from_text(self, text: str):
        """自动从文本中提取并更新状态"""
        extracted = self.extract_from_text(text)
        for state in extracted:
            self.add_state(state)
        return extracted

    def get_active_states(self) -> List[Dict[str, Any]]:
        """Get all currently valid states (not expired)"""
        self.cleanup_expired_states()
        # Return list of states sorted by creation time
        return sorted(self.states.values(), key=lambda x: x["created_at"])

    def get_context_string(self) -> str:
        """
        Get a formatted string of current states for LLM context.
        Example:
        [Current Status]
        - ate dinner (Completed)
        - take medicine (Active)
        """
        states = self.get_active_states()
        if not states:
            return ""
            
        lines = ["[Persistent States / Daily Actions]"]
        for state in states:
            # Format time if needed, but relative status is more important
            status = state["status"]
            content = state["content"]
            lines.append(f"- {content} [{status}]")
            
        return "\n".join(lines)
