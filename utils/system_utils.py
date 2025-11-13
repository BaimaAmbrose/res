# utils/system_utils.py
from errno import EEXIST
from os import makedirs, path
import os
import re


def mkdir_p(folder_path: str):
    """
    递归创建目录；存在则忽略。
    """
    try:
        makedirs(folder_path)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder: str) -> int:
    """
    在给定目录下查找最大迭代号。
    兼容：
      - 目录名：iteration_XXXX
      - 文件名：chkpntXXXX.pth
    若不存在匹配项，返回 0。
    """
    if not path.isdir(folder):
        return 0

    max_iter = 0
    for name in os.listdir(folder):
        # 1) 目录：iteration_XXXX
        if name.startswith("iteration_"):
            tail = name[len("iteration_"):]
            if tail.isdigit():
                max_iter = max(max_iter, int(tail))
                continue

        # 2) 文件：chkpntXXXX.pth
        m = re.match(r"^chkpnt(\d+)\.pth$", name)
        if m:
            max_iter = max(max_iter, int(m.group(1)))
            continue

        # 3) 其它命名里最后一个连续数字作为兜底（可选）
        m = re.search(r"(\d+)", name)
        if m:
            # 如不想兜底，可删除此分支
            max_iter = max(max_iter, int(m.group(1)))

    return max_iter
