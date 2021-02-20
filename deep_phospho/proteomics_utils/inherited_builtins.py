import inspect, re


class NonOverwriteDict(dict):
    def __setitem__(self, key, value):
        if self.__contains__(key):
            pass
        else:
            dict.__setitem__(self, key, value)


def varname(var):
    """
    https://stackoverflow.com/questions/592746/how-can-you-print-a-variable-name-in-python
    通过调用这个函数时traceback得到code_content，re到需要的var name
    Traceback(filename='<ipython-input-37-5fa84b05d0d4>', lineno=2, function='<module>', code_context=['b = varname(a)\n'], index=0)
    拿到varname(...)里的内容
    """
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
        return m.group(1)

