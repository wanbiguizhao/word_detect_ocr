import paddle 
from paddle.io import Dataset
import cv2 as cv
#一种思路是加载一个obj信息。
class MLPDataset(Dataset):
    def __init__(self,labels_image_info,transform=None):
        super().__init__()
        #self.image_list=list(map(lambda x:cv.resize(cv.imread(x,cv.IMREAD_GRAYSCALE),[64,16]),labels_image_info["Image_Path"] ))
        self.image_list=list(map(lambda x:cv.imread(x,cv.IMREAD_GRAYSCALE),labels_image_info["Image_Path"] ))
        self.image_type_list=labels_image_info["Image_Type"]
        self.image_import_flag_list=labels_image_info["Import_Flag"]
        self.transform=transform

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        image = self.image_list[index]
        float_image=image.astype('float32')
        image_type=self.image_type_list[index]
        image_import_flag=self.image_import_flag_list[index]

        if self.transform is not None:
            image_t = self.transform(float_image)
            return [image_t,image], image_type,image_import_flag
        return [image,image], image_type,image_import_flag

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.image_list)
