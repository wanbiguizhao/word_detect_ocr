# 把图片的所有图片都组合起来，方便人工校验，是否可以切割图片。
import os
import re
PROJECT_DIR= os.path.dirname(
    os.path.realpath( __file__)
)
from  config import env_template as template

def render_html(word_image_slice_dir):
    #word_image_slice_dir 存放图像切片的目录。
    image_info_table=[]
    row_info=[]
    col_count=0
    print(os.getcwd())
    for image_info in sorted(os.listdir(word_image_slice_dir)):
        if "png" not in image_info:
            continue
        if len(row_info)>40:
            image_info_table.append(row_info)
            row_info=[]
        
        matchObj = re.match(
            r'word.*_(?P<id_ds>\d+)_.*_(?P<id_cluster>\d+)', image_info) # 图片切割的文件名格式
        #print(matchObj)
        if not matchObj:
            continue
        id_ds=matchObj.groupdict()["id_ds"]
        id_cluster=matchObj.groupdict()["id_cluster"]
        row_info.append(
            {
                "path":image_info,
                "id_ds":id_ds[1:],
                "id_cluster":id_cluster
            }
        )
    image_info_table.append(row_info)
    #print(image_info_table)
    html_data=template.render(image_info_table=image_info_table)
    with open("{word_image_slice_dir}/imagetable.html",'w') as ith:
        ith.write(html_data)