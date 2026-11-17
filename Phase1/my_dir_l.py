import os
import time

def my_dir_l(path="."):
    # 遍历当前目录下所有条目（不递归子目录）
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        stat_info = os.stat(full_path)

        # 文件类型：目录d / 文件-
        file_type = "d" if os.path.isdir(full_path) else "-"
        # 简单权限（简化版，只拿user读写执行）
        mode = stat_info.st_mode
        perm = ""
        perm += "r" if mode & 0o400 else "-"
        perm += "w" if mode & 0o200 else "-"
        perm += "x" if mode & 0o100 else "-"

        size = stat_info.st_size
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat_info.st_mtime))
        print(f"{file_type}{perm} {size:8d} {mtime}  {name}")


def find_file(keyword):
    # os.walk 递归遍历
    for root, dirs, files in os.walk("."):
        for filename in files:
            if keyword in filename:
                # 拼接完整路径，再转成相对路径
                full = os.path.join(root, filename)
                rel_path = os.path.relpath(full, ".")
                print(rel_path)

if __name__ == "__main__":
    my_dir_l()
    target = input("请输入要查找的文件名关键字：")
    find_file(target)
