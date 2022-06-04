import wx

from deep_phospho.desktop_app.ui_wx import init_frame

if __name__ == '__main__':
    app_main = wx.App()  # redirect=True, filename=WorkDir.get_log_path())
    # screen_size = wx.DisplaySize()
    # frame_pos = (screen_size[0] / 8, screen_size[1] / 15)
    init_frame()
    app_main.MainLoop()
