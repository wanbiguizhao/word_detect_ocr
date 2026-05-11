import json
from pathlib import Path
import cv2
import numpy as np

class PseudoLabelGenerator:
    def __init__(self, config):
        self.config = config
        self.dataset_id = config.get("dataset.current", "pdf5823")
        self.high_confidence_threshold = config.get("labeling.high_confidence_threshold", 0.9)
        self.low_confidence_threshold = config.get("labeling.low_confidence_threshold", 0.7)
        
        self.project_root = Path(__file__).parent.parent.parent
        self.datahome_dir = self.project_root / "bussiness" / "datahome"
        self.dataset_dir = self.datahome_dir / self.dataset_id

        self.prelabels_path = self.dataset_dir / "pre_labels.json"
        self.unified_labels_path = self.dataset_dir / "unified_labels.json"

        if not self.prelabels_path.exists():
            self.prelabels_path = self.project_root / "bussiness" / "pre_labels.json"
        if not self.unified_labels_path.exists():
            self.unified_labels_path = self.project_root / "bussiness" / "unified_labels.json"
        
        self.ocr_model = None
        self.char_mapping = None

    def load_ocr_model(self):
        try:
            import torch
            from ocr_system.models.ocr_model import OCRModel
            
            model_path = self.project_root / self.config.get("ocr.model_path", "ocr_system/model_store/inference/ocr_model_best.pth")
            self.ocr_model = OCRModel()
            self.ocr_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.ocr_model.eval()
            
            from ocr_system.configs.char_mapping import CharMappingManager
            self.char_mapping = CharMappingManager()
            
            print("✅ OCR模型加载成功")
        except Exception as e:
            print(f"❌ OCR模型加载失败: {e}")
            raise

    def predict(self, image_path):
        if self.ocr_model is None:
            self.load_ocr_model()
        
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None, 0.0
            
            img = cv2.resize(img, (64, 64))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=0)
            
            import torch
            with torch.no_grad():
                output = self.ocr_model(torch.tensor(img, dtype=torch.float32))
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)
            
            predicted_char = self.char_mapping.get_char(predicted_idx.item())
            
            return predicted_char, confidence.item()
        
        except Exception as e:
            print(f"⚠️  预测失败 {image_path}: {e}")
            return None, 0.0

    def get_unlabeled_images(self):
        if self.unified_labels_path.exists():
            unified_labels_path = self.unified_labels_path
        else:
            unified_labels_path = self.project_root / "bussiness" / "unified_labels.json"

        if not unified_labels_path.exists():
            print("❌ unified_labels.json 不存在")
            return []

        with open(unified_labels_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        unlabeled_images = []
        for ann in data.get("annotations", []):
            if ann.get("dataset") == self.dataset_id:
                if ann.get("status") != "labeled":
                    img_path = self.project_root / "bussiness" / ann["image_path"]
                    if img_path.exists():
                        unlabeled_images.append({
                            "image_path": str(img_path.relative_to(self.project_root / "bussiness")),
                            "char_id": ann.get("char_id", "")
                        })

        return unlabeled_images

    def generate_pseudo_labels(self):
        print(f"📊 开始生成预标注 - 数据集: {self.dataset_id}")
        
        images = self.get_unlabeled_images()
        print(f"🔍 找到 {len(images)} 张未标注图片")
        
        if len(images) == 0:
            print("⚠️  没有未标注图片")
            return
        
        print("🔄 加载OCR模型...")
        self.load_ocr_model()
        
        print("🔄 执行预测...")
        prelabels = []
        high_conf_count = 0
        low_conf_count = 0
        
        for img_info in images:
            full_path = self.project_root / "bussiness" / img_info["image_path"]
            predicted_char, confidence = self.predict(full_path)
            
            if predicted_char:
                if confidence >= self.high_confidence_threshold:
                    level = "high"
                    high_conf_count += 1
                elif confidence >= self.low_confidence_threshold:
                    level = "medium"
                else:
                    level = "low"
                    low_conf_count += 1
                
                prelabels.append({
                    "image_path": img_info["image_path"],
                    "char_id": img_info["char_id"],
                    "predicted_char": predicted_char,
                    "confidence": round(confidence, 4),
                    "confidence_level": level,
                    "status": "pending"
                })
        
        result = {
            "version": "1.0",
            "dataset": self.dataset_id,
            "generated_at": "2024-01-15T10:30:00",
            "stats": {
                "total": len(prelabels),
                "high_confidence": high_conf_count,
                "low_confidence": low_conf_count
            },
            "prelabels": prelabels
        }
        
        with open(self.prelabels_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 预标注完成！")
        print(f"   总数: {len(prelabels)}")
        print(f"   高置信度: {high_conf_count}")
        print(f"   低置信度: {low_conf_count}")
        
        return result

if __name__ == "__main__":
    from server.backend.config import config
    generator = PseudoLabelGenerator(config._config)
    generator.generate_pseudo_labels()