"""
通用的spectral library格式
只负责储存信息，不同来源的library从这里继承
预留一个不同library的extra info保存位置

包含以下属性
以peptide为key的RT dict
以precursor为key的inten dict
以protein为key的description dict
保存的总peptide、precursor、protein


包含以下方法
读取library（pickle，json）
不同来源的读取在子类中写
储存（pickle，json）
查询：查询一个蛋白的所有信息
    一个precursor的所有信息
    一个peptide的所有信息
    RT在一定范围的peptide
    RT在一定范围的peptide对应的precurosr和protein -> peptide和precursor和protein之间互相的关系
合并library（以其中一个library的protein为主，precursor可以完全替换或选fragment多的替换或以一边为主增加额外）
合并library（选两边library中precursor或peptide多的protein）

增加或删除protein
增加或删除precursor
增加或删除peptide

统计protein（数量）
统计precursor（数量、长度分布、aa分布、charge分布） -> 这里的precursor数量等的统计不删除重复的peptide
统计peptide（数量、长度分布、aa分布）
统计RT



"""

