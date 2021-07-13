import wx

from deep_phospho.desktop_app.ui_wx import DeepPhosphoUIFrame

if __name__ == '__main__':
    app_main = wx.App()  # redirect=True, filename=WorkDir.get_log_path())
    screen_size = wx.DisplaySize()
    pos_middle = (screen_size[0] / 8, screen_size[1] / 15)
    frame_search = DeepPhosphoUIFrame(None, pos=pos_middle, title='DeepPhospho', size=(1200, 800))
    frame_search.Show(True)
    app_main.MainLoop()
