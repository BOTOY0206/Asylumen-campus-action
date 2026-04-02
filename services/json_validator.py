from typing import Dict
import jsonschema

JSON_SCHEMA = {
    "type": "object",
    "required": ["task_id", "status", "video_path", "output_path", "error_message", "result"],
    "properties": {
        "task_id": {"type": "string"},
        "status": {"type": "string", "enum": ["success", "failed", "running"]},
        "video_path": {"type": "string"},
        "output_path": {"type": "string"},
        "error_message": {"type": "string"},
        "result": {
            "type": "object",
            "required": ["frames"],
            "properties": {
                "frames": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["frame_id", "timestamp", "bbox", "behavior", "confidence"],
                        "properties": {
                            "frame_id": {"type": "integer"},
                            "timestamp": {"type": "number"},
                            "bbox": {"type": "array", "items": {"type": "number"}},
                            "behavior": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                }
            }
        }
    }
}

def validate_json(data: Dict) -> bool:
    try:
        jsonschema.validate(instance=data, schema=JSON_SCHEMA)
        return True
    except jsonschema.ValidationError as e:
        print(f"JSON校验失败: {e}")
        return False

def save_standard_json(data: Dict, output_path: str) -> None:
    import json
    if not validate_json(data):
        raise ValueError("JSON格式不符合规范")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
