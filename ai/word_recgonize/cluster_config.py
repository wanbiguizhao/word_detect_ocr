"""
聚类配置文件
定义HOG聚类的参数和输出路径
"""

from dataclasses import dataclass
from pathlib import Path

@dataclass
class HOGClusterConfig:
    """
    HOG特征聚类配置
    
    属性:
        input_image_dir: 输入汉字图片目录
        lineage_file: 血缘关系文件路径
        output_dir: 聚类结果输出目录
        clusters_json: 聚类结果JSON文件名
        cluster_images_dir: 聚类示例图片目录名
        
        # HOG参数
        orientations: HOG方向数量
        pixels_per_cell: 每个细胞的像素数
        cells_per_block: 每个块的细胞数
        
        # 聚类参数
        n_clusters: 聚类数量
        random_state: 随机种子
    """
    # 输入路径
    input_image_dir: str = "bussiness/datahome/pdf01/pdf_chars"
    lineage_file: str = "bussiness/datahome/pdf01/lineage.json"
    
    # 输出路径
    output_dir: str = "bussiness/datahome/pdf01/clusters"
    clusters_json: str = "hog_clusters.json"
    cluster_images_dir: str = "cluster_images"
    
    # HOG参数
    orientations: int = 9
    pixels_per_cell: tuple = (8, 8)
    cells_per_block: tuple = (2, 2)
    
    # 聚类参数
    n_clusters: int = 50
    random_state: int = 42
    
    def get_input_path(self) -> Path:
        """获取输入图片目录的绝对路径"""
        return Path(__file__).resolve().parent.parent.parent / self.input_image_dir
    
    def get_lineage_path(self) -> Path:
        """获取血缘关系文件的绝对路径"""
        return Path(__file__).resolve().parent.parent.parent / self.lineage_file
    
    def get_output_path(self) -> Path:
        """获取输出目录的绝对路径"""
        return Path(__file__).resolve().parent.parent.parent / self.output_dir
    
    def get_clusters_json_path(self) -> Path:
        """获取聚类结果JSON文件路径"""
        return self.get_output_path() / self.clusters_json
    
    def get_cluster_images_path(self) -> Path:
        """获取聚类示例图片目录路径"""
        return self.get_output_path() / self.cluster_images_dir