"""
PDF数据集评估器
支持评估标注数据和所有数据
"""
import os
import json
from collections import Counter

from .evaluator import Evaluator


class PDFEvaluator(Evaluator):
    """PDF数据集评估器"""
    
    def __init__(self, data_dir, model_path=None):
        """
        初始化PDF评估器
        
        Args:
            data_dir: PDF数据集目录（如 pdf5823, pdf01）
            model_path: 模型路径
        """
        super().__init__(model_path)
        self.data_dir = data_dir
        
        # 路径配置
        self.cluster_file = os.path.join(data_dir, 'clusters', 'hog_clusters.json')
        self.label_file = os.path.join(data_dir, 'clusters', 'labeling', 'labels.json')
        self.rule_jsons_dir = os.path.join(data_dir, 'rule_jsons')
        self.pdf_lines_dir = os.path.join(data_dir, 'pdf_lines')
    
    def load_labeled_data(self):
        """加载已标注的数据"""
        if not os.path.exists(self.label_file):
            raise FileNotFoundError(f"标注文件不存在: {self.label_file}")
        
        if not os.path.exists(self.cluster_file):
            raise FileNotFoundError(f"Cluster文件不存在: {self.cluster_file}")
        
        # 读取标注
        with open(self.label_file, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        
        # 读取cluster
        with open(self.cluster_file, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        
        # 获取已标注的字符图像
        image_paths = []
        true_chars = []
        char_ids = []
        
        for cluster_id, label_info in labels_data.items():
            if label_info.get('status') != 'labeled' or label_info.get('char') is None:
                continue
            
            true_char = label_info['char']
            
            if cluster_id not in cluster_data.get('clusters', {}):
                continue
            
            for char_info in cluster_data['clusters'][cluster_id]:
                image_path = char_info['image_path']
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    true_chars.append(true_char)
                    char_ids.append(char_info['char_id'])
        
        return {
            'image_paths': image_paths,
            'true_chars': true_chars,
            'char_ids': char_ids,
            'total_labeled': len(image_paths)
        }
    
    def load_all_data(self):
        """加载所有数据（标注+未标注）"""
        if not os.path.exists(self.cluster_file):
            raise FileNotFoundError(f"Cluster文件不存在: {self.cluster_file}")
        
        with open(self.cluster_file, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        
        # 获取所有字符图像
        all_data = []
        
        for cluster_id, chars in cluster_data.get('clusters', {}).items():
            for char_info in chars:
                image_path = char_info['image_path']
                if os.path.exists(image_path):
                    all_data.append({
                        'image_path': image_path,
                        'char_id': char_info['char_id'],
                        'cluster_id': cluster_id
                    })
        
        return {
            'data': all_data,
            'total': len(all_data)
        }
    
    def evaluate_labeled(self):
        """评估已标注的数据"""
        print(f"加载已标注数据...")
        labeled_data = self.load_labeled_data()
        print(f"已标注图像数: {labeled_data['total_labeled']}")
        
        if labeled_data['total_labeled'] == 0:
            print("没有已标注的数据")
            return None
        
        results = self.evaluate_with_ground_truth(
            labeled_data['image_paths'],
            labeled_data['true_chars']
        )
        
        # 添加额外信息
        results['char_ids'] = labeled_data['char_ids']
        
        return results
    
    def evaluate_all(self):
        """评估所有数据（标注+未标注）"""
        print(f"加载所有数据...")
        all_data = self.load_all_data()
        print(f"总图像数: {all_data['total']}")
        
        if all_data['total'] == 0:
            print("没有数据")
            return None
        
        # 获取图像路径列表
        image_paths = [item['image_path'] for item in all_data['data']]
        
        # 批量预测
        pred_results = self.batch_predict(image_paths)
        
        # 合并结果
        results = {
            'total': all_data['total'],
            'results': []
        }
        
        for i, item in enumerate(all_data['data']):
            pred = pred_results[i]
            results['results'].append({
                'char_id': item['char_id'],
                'cluster_id': item['cluster_id'],
                'image_path': item['image_path'],
                'pred_char': pred['pred_char'],
                'confidence': pred['confidence'],
                'label_id': pred.get('label_id'),
                'error': pred.get('error')
            })
        
        # 统计预测分布
        pred_distribution = Counter()
        for r in results['results']:
            if r['pred_char']:
                pred_distribution[r['pred_char']] += 1
        results['pred_distribution'] = dict(pred_distribution)
        
        return results
    
    def evaluate_with_mixed(self):
        """
        评估所有数据，同时标记哪些是已标注的
        返回包含标注信息的完整评估结果
        """
        print(f"加载所有数据...")
        
        # 加载标注数据
        labeled_map = {}
        if os.path.exists(self.label_file):
            with open(self.label_file, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
            
            for cluster_id, label_info in labels_data.items():
                if label_info.get('status') == 'labeled' and label_info.get('char') is not None:
                    labeled_map[cluster_id] = label_info['char']
        
        # 加载所有数据
        if not os.path.exists(self.cluster_file):
            raise FileNotFoundError(f"Cluster文件不存在: {self.cluster_file}")
        
        with open(self.cluster_file, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        
        # 准备数据
        image_paths = []
        true_chars = []
        char_info_list = []
        
        for cluster_id, chars in cluster_data.get('clusters', {}).items():
            true_char = labeled_map.get(cluster_id)
            
            for char_info in chars:
                image_path = char_info['image_path']
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    true_chars.append(true_char)  # None 表示未标注
                    char_info_list.append({
                        'char_id': char_info['char_id'],
                        'cluster_id': cluster_id,
                        'is_labeled': (true_char is not None)
                    })
        
        print(f"总图像数: {len(image_paths)}, 已标注: {len(labeled_map)}个cluster")
        
        # 批量预测
        results = []
        pred_distribution = Counter()
        labeled_correct = 0
        labeled_total = 0
        
        for i, (image_path, true_char) in enumerate(tqdm(zip(image_paths, true_chars), desc='Evaluating')):
            try:
                pred_char, confidence, label_id = self.predict_single(image_path)
                
                is_correct = None
                if true_char is not None:
                    labeled_total += 1
                    is_correct = (pred_char == true_char)
                    if is_correct:
                        labeled_correct += 1
                
                results.append({
                    'char_id': char_info_list[i]['char_id'],
                    'cluster_id': char_info_list[i]['cluster_id'],
                    'image_path': image_path,
                    'true_char': true_char,
                    'pred_char': pred_char,
                    'confidence': confidence,
                    'label_id': label_id,
                    'is_labeled': char_info_list[i]['is_labeled'],
                    'is_correct': is_correct
                })
                
                pred_distribution[pred_char] += 1
            
            except Exception as e:
                results.append({
                    'char_id': char_info_list[i]['char_id'],
                    'cluster_id': char_info_list[i]['cluster_id'],
                    'image_path': image_path,
                    'true_char': true_char,
                    'pred_char': None,
                    'confidence': 0.0,
                    'label_id': None,
                    'is_labeled': char_info_list[i]['is_labeled'],
                    'is_correct': None,
                    'error': str(e)
                })
        
        labeled_accuracy = labeled_correct / labeled_total if labeled_total > 0 else 0.0
        
        return {
            'total': len(results),
            'labeled_total': labeled_total,
            'labeled_correct': labeled_correct,
            'labeled_accuracy': labeled_accuracy,
            'results': results,
            'pred_distribution': dict(pred_distribution)
        }