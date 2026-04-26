import os
from pathlib import Path
import filecmp
import time

def compare_by_name(dir_src: Path, dir_target: Path):
    """
    比较文件名：判断 src 中的文件名是否在 target 中出现过
    返回：同名文件名字列表
    """
    print("="*50)
    print("模式一：按【文件名】比较")
    print("="*50)
    
    # 获取 B 文件夹中所有的文件名，构建一个集合 (查找速度 O(1))
    b_files = {f.name for f in dir_target.iterdir() if f.is_file()}
    
    found_count = 0
    not_found_count = 0
    same_name_files = []  # 存储同名文件名
    
    for file_a in dir_src.iterdir():
        if file_a.is_file():
            if file_a.name in b_files:
                print(f"✅ [找到同名] {dir_src.name}中的 {file_a.name} 在 {dir_target.name} 中存在")
                found_count += 1
                same_name_files.append(file_a.name)  # 记录同名文件
    
    print(f"\n统计: 共检查 {found_count + not_found_count} 个文件。找到 {found_count} 个，未找到 {not_found_count} 个。\n")
    return same_name_files  # 返回同名列表

def compare_by_content(dir_src: Path, dir_b: Path):
    """
    比较文件内容：判断 A 中的文件内容是否和 B 中的某个文件完全一致
    即使文件名不同，只要内容一致也会被匹配出来
    """
    print("="*50)
    print("模式二：按【文件内容】比较 (深度比较，忽略文件名)")
    print("="*50)
    
    # 获取 B 文件夹中所有的文件对象列表
    b_files = [f for f in dir_b.iterdir() if f.is_file()]
    
    matched_count = 0
    unmatched_count = 0
    
    for file_a in dir_src.iterdir():
        if file_a.is_file():
            is_matched = False
            matched_name = ""
            
            # 将 A 中的每个文件与 B 中的每个文件比较内容
            for file_b in b_files:
                # filecmp.cmp 默认 shallow=False 时会比较文件大小和内容，非常高效
                if filecmp.cmp(file_a, file_b, shallow=False):
                    is_matched = True
                    matched_name = file_b.name
                    break # 找到一个内容相同的就跳出内层循环
                    
            if is_matched:
                print(f"✅ [内容一致] A中的 {file_a.name} 与 B中的 {matched_name} 内容完全相同")
                matched_count += 1
            else:
                print(f"❌ [内容不同] A中的 {file_a.name} 在 B 中没有内容相同的文件")
                unmatched_count += 1
                
    print(f"\n统计: 共检查 {matched_count + unmatched_count} 个文件。内容匹配 {matched_count} 个，不匹配 {unmatched_count} 个。\n")

def rename_same_files(target_dir: Path, same_name_list: list):
    """
    对目标文件夹中【同名文件】批量重命名
    规则：xdel_原文件名
    """
    if not same_name_list:
        print("ℹ️ 没有需要重命名的同名文件\n")
        return

    print("="*50)
    print("开始执行【同名文件重命名】")
    print(f"重命名规则：xdel_原文件名")
    print(f"目标文件夹：{target_dir.name}")
    print("="*50)

    success_count = 0
    fail_count = 0

    for file_name in same_name_list:
        file_path = target_dir / file_name
        if file_path.exists() and file_path.is_file():
            # 新文件名：xdel_原文件名
            new_name = f"xdel_{file_name}"
            new_path = target_dir / new_name

            try:
                file_path.rename(new_path)
                print(f"✅ 重命名成功：{file_name} → {new_name}")
                success_count += 1
            except Exception as e:
                print(f"❌ 重命名失败：{file_name}，错误：{str(e)}")
                fail_count += 1
        else:
            print(f"⚠️ 文件不存在：{file_name}")
            fail_count += 1

    print(f"\n重命名完成：成功 {success_count} 个，失败 {fail_count} 个\n")


def clear_dirty_file():
    """
    保持done文件的干净，会把unlable的文件夹的文件和done的文件进行对比，如果done中已经标记过，就对unlable中的数据进行清除。
    """
    # 1. 获取当前脚本的绝对路径
    script_path = Path(__file__).resolve()
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    print(f"当前脚本路径: {script_path}")
    
    # 2. 获取当前脚本所在的目录
    script_dir = script_path.parent
    print(f"脚本所在目录: {script_dir}\n")
    
    # 3. 定义文件夹路径（可自行修改）
    dir_src = script_dir / "done2"       # 源文件夹
    dir_target = PROJECT_ROOT/ "images" / "filtered_images"  # 目标文件夹（要重命名的文件夹）
    
    # 4. 检查文件夹是否存在（修复了原代码判断BUG）
    if not dir_src.exists() or not dir_src.is_dir():
        print(f"❌ 错误: 找不到源文件夹：{dir_src}")
        exit(1)
    if not dir_target.exists() or not dir_target.is_dir():
        print(f"❌ 错误: 找不到目标文件夹：{dir_target}")
        exit(1)
        
    print(f"源文件夹 dir_src 路径: {dir_src}")
    print(f"目标文件夹 dir_target 路径: {dir_target}\n")
    
    # 5. 执行文件名比较，获取同名文件列表
    try:
        same_files = compare_by_name(dir_src, dir_target)
        
        # 6. 对目标文件夹执行重命名
        rename_same_files(dir_target, same_files)
        
        # 可选：继续执行内容比较
        # compare_by_content(dir_src, dir_target)
        
    except Exception as e:
        print(f"程序发生错误: {e}")   


def process_json_files(done_dir: Path, file_suffix: str = '_fusion.json'):
    """
    处理done文件夹中的json文件：
    1. 查找所有以_chars_marked.json结尾的文件
    2. 检查重命名后是否存在文件名冲突
    3. 显示统计信息，等待5秒后执行批量替换
    4. 如果有冲突，5秒后退出
    """
    print("="*50)
    print("开始处理 JSON 文件重命名")
    print(f"目标文件夹：{done_dir}")
    print("="*50)
    
    # 查找所有以_chars_marked.json结尾的文件
    json_files = [f for f in done_dir.iterdir() if f.is_file() and f.name.endswith(file_suffix)]
    
    if not json_files:
        print(f"ℹ️ 没有找到需要处理的 {file_suffix} 文件\n")
        return
    
    # 准备重命名映射
    rename_map = []  # [(原路径, 新路径, 新文件名), ...]
    conflict_files = []  # 有冲突的文件
    
    for json_file in json_files:
        # 新文件名：去掉 _chars_marked.json，加上 .json
        new_name = json_file.name.replace(file_suffix, '.json')
        new_path = done_dir / new_name
        
        # 检查重命名后是否会与现有文件冲突
        if new_path.exists() and new_path != json_file:
            conflict_files.append((json_file.name, new_name))
        else:
            rename_map.append((json_file, new_path, new_name))
    
    # 显示统计信息
    print(f"\n📊 统计信息：")
    print(f"   找到 {file_suffix} 文件: {len(json_files)} 个")
    print(f"   可以重命名: {len(rename_map)} 个")
    print(f"   文件名冲突: {len(conflict_files)} 个")
    
    if conflict_files:
        print(f"\n⚠️ 以下文件重命名后会产生冲突：")
        for old_name, new_name in conflict_files:
            print(f"   {old_name} → {new_name} (已存在)")
        print(f"\n⏳ 5秒后退出程序...")
        time.sleep(5)
        return
    
    if not rename_map:
        print("\nℹ️ 没有需要重命名的文件")
        return
    
    print(f"\n✅ 检查通过，没有发现文件名冲突")
    print(f"⏳ 5秒后开始执行批量重命名...")
    time.sleep(5)
    
    # 执行批量重命名
    print(f"\n开始执行重命名...")
    success_count = 0
    fail_count = 0
    
    for old_path, new_path, new_name in rename_map:
        try:
            old_name = old_path.name
            old_path.rename(new_path)
            print(f"✅ {old_name} → {new_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ 重命名失败：{old_path.name}，错误：{str(e)}")
            fail_count += 1
    
    print(f"\n重命名完成：成功 {success_count} 个，失败 {fail_count} 个\n")


if __name__ == "__main__":
    # 获取当前脚本的绝对路径
    script_path = Path(__file__).resolve()
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    print(f"当前脚本路径: {script_path}")
    
    # 获取当前脚本所在的目录
    script_dir = script_path.parent
    print(f"脚本所在目录: {script_dir}\n")
    
    # 定义done文件夹路径
    done_dir = script_dir / "done2"
    
    # 检查done文件夹是否存在
    if not done_dir.exists() or not done_dir.is_dir():
        print(f"❌ 错误: 找不到done文件夹：{done_dir}")
        exit(1)
    
    # 先处理json文件重命名
    process_json_files(done_dir, file_suffix='fusion.json')
    
    # 然后执行原有的清理逻辑
    # clear_dirty_file()
