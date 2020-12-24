"""
一个result类同时包含RT和intensity，但是可以获得不同的file并选定file获得指定数据，可以对同一个数据创建两个result的实例然后各自存储

针对每一个结果定义title name constant

从base result继承两个分别用于RT和intensity的类
"""


class SearchResult(object):
    def __init__(self):
        self.result_df = None

    def __getitem__(self, item):
        return self.result_df[item]
