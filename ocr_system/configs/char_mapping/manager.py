import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple


class CharMappingManager:
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            self.config_dir = Path(__file__).parent
        else:
            self.config_dir = Path(config_dir)

        self.char_to_label_path = self.config_dir / "char_to_label.json"
        self.label_to_char_path = self.config_dir / "label_to_char.json"
        self.char_properties_path = self.config_dir / "char_properties.json"
        self.dataset_mapping_path = self.config_dir / "dataset_mapping.json"

        self._load_configs()

    def _load_configs(self):
        with open(self.char_to_label_path, "r", encoding="utf-8") as f:
            self.char_to_label = json.load(f)

        with open(self.label_to_char_path, "r", encoding="utf-8") as f:
            self.label_to_char = json.load(f)

        with open(self.char_properties_path, "r", encoding="utf-8") as f:
            self.char_properties = json.load(f)

        with open(self.dataset_mapping_path, "r", encoding="utf-8") as f:
            self.dataset_mapping = json.load(f)

    def _save_configs(self):
        with open(self.char_to_label_path, "w", encoding="utf-8") as f:
            json.dump(self.char_to_label, f, ensure_ascii=False, indent=2)

        with open(self.label_to_char_path, "w", encoding="utf-8") as f:
            json.dump(self.label_to_char, f, ensure_ascii=False, indent=2)

        with open(self.char_properties_path, "w", encoding="utf-8") as f:
            json.dump(self.char_properties, f, ensure_ascii=False, indent=2)

        with open(self.dataset_mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset_mapping, f, ensure_ascii=False, indent=2)

    def get_label_id(self, char: str) -> Optional[int]:
        if char in self.char_to_label["characters"]:
            return self.char_to_label["characters"][char]["label_id"]
        if char in self.char_to_label["custom_characters"]:
            return self.char_to_label["custom_characters"][char]["label_id"]
        return None

    def get_char(self, label_id: int) -> Optional[str]:
        return self.label_to_char["labels"].get(str(label_id), {}).get("char")

    def add_custom_char(self, char: str, pinyin: str = "", radical: str = "",
                        stroke_count: int = 0, unicode: str = "") -> Tuple[bool, int]:
        if char in self.char_to_label["characters"]:
            return False, self.char_to_label["characters"][char]["label_id"]

        if char in self.char_to_label["custom_characters"]:
            return False, self.char_to_label["custom_characters"][char]["label_id"]

        label_id = self.char_to_label["next_custom_id"]

        self.char_to_label["custom_characters"][char] = {
            "label_id": label_id,
            "added_in_version": self.char_to_label["version"],
            "status": "custom"
        }
        self.char_to_label["next_custom_id"] += 1
        self.char_to_label["last_updated"] = datetime.now().isoformat()

        self.label_to_char["labels"][str(label_id)] = {
            "char": char,
            "status": "custom"
        }

        if char not in self.char_properties["characters"]:
            self.char_properties["characters"][char] = {
                "pinyin": pinyin,
                "radical": radical,
                "stroke_count": stroke_count,
                "unicode": unicode
            }

        self._save_configs()
        return True, label_id

    def batch_add_chars(self, chars: list) -> Dict[str, int]:
        result = {}
        for char_info in chars:
            char = char_info["char"]
            pinyin = char_info.get("pinyin", "")
            radical = char_info.get("radical", "")
            stroke_count = char_info.get("stroke_count", 0)
            unicode = char_info.get("unicode", "")

            _, label_id = self.add_custom_char(char, pinyin, radical, stroke_count, unicode)
            result[char] = label_id
        return result

    def get_char_info(self, char: str) -> Optional[Dict]:
        info = {}
        if char in self.char_to_label["characters"]:
            info.update(self.char_to_label["characters"][char])
        elif char in self.char_to_label["custom_characters"]:
            info.update(self.char_to_label["custom_characters"][char])
        else:
            return None

        if char in self.char_properties["characters"]:
            info.update(self.char_properties["characters"][char])

        return info

    def get_all_chars(self) -> Dict[str, Dict]:
        result = {}
        result.update(self.char_to_label["characters"])
        result.update(self.char_to_label["custom_characters"])
        return result

    def get_stats(self) -> Dict:
        return {
            "version": self.char_to_label["version"],
            "charset": self.char_to_label["charset"],
            "standard_chars_count": len(self.char_to_label["characters"]),
            "custom_chars_count": len(self.char_to_label["custom_characters"]),
            "total_chars_count": len(self.char_to_label["characters"]) + len(self.char_to_label["custom_characters"]),
            "next_custom_id": self.char_to_label["next_custom_id"],
            "last_updated": self.char_to_label["last_updated"]
        }


if __name__ == "__main__":
    manager = CharMappingManager()

    print("=== 统计信息 ===")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n=== 获取字符ID ===")
    print(f"中: {manager.get_label_id('中')}")
    print(f"笙: {manager.get_label_id('笙')}")

    print("\n=== 获取汉字 ===")
    print(f"ID 0: {manager.get_char(0)}")
    print(f"ID 6863: {manager.get_char(6863)}")

    print("\n=== 添加新字符 ===")
    success, label_id = manager.add_custom_char("赟", "yūn", "贝", 16, "U+8D5F")
    char_str = "赟" if success else "已存在"
    print(f"添加 {char_str}, ID: {label_id}")

    print("\n=== 新统计信息 ===")
    stats = manager.get_stats()
    print(f"总字符数: {stats['total_chars_count']}")