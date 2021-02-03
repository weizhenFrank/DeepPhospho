import os


def change_cmd_page(page=65001):
    """
    65001 -> utf-8
    936 -> gbk
    """
    os.system(f'chcp {page}')
