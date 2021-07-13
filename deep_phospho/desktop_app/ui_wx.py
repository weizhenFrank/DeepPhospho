# import webbrowser
import datetime
import os

import ipdb
import wx
import wx.richtext

from .runner_for_ui import parse_args_from_ui_to_runner, RunnerThread
from .ui_configs import *

# Defined params
PipelineParams = {
    'WorkFolder': '.',
    'TaskName': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    'Pretrain-Ion': '',
    'Pretrain-RT-4': '',
    'Pretrain-RT-5': '',
    'Pretrain-RT-6': '',
    'Pretrain-RT-7': '',
    'Pretrain-RT-8': '',
    'RTEnsemble': True,
    'Device': '0',
    'RTScale-lower': '-100',
    'RTScale-upper': '200',
    'MaxPepLen': '54',
    'TrainData': '',
    'TrainDataFormat': '',
    'Epoch-Ion': '30',
    'Epoch-RT': '30',
    'BatchSize-Ion': '64',
    'BatchSize-RT': '128',
    'InitLR': '0.0001',
    'PredInput': [],
    'PredInputFormat': [],
    'train':False,
    'pred':False,
}


class DeepPhosphoUIFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(DeepPhosphoUIFrame, self).__init__(*args, **kwargs)

        self._main_panel = wx.Panel(self, style=wx.TAB_TRAVERSAL | wx.CLIP_CHILDREN | wx.FULL_REPAINT_ON_RESIZE)

        self._create_menubar()

        # Status control
        self._running_status = False
        self._status_bar = self._init_status_bar()

        # Font
        self._font_boxname = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False)
        self._font_static_text = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False)
        self._font_search_content = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False)
        self._font_listbox = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_LIGHT, False)

        # Total sizer
        self.boxsizer_main = wx.BoxSizer(wx.VERTICAL)  # Contain common and external tool

        """ Description text """
        desc_text_staticboxsizer = self._init_desc_text()
        self.boxsizer_main.Add(desc_text_staticboxsizer, 0, wx.ALL, 10)

        """ Pipeline """
        pipeline_boxsizer = self._init_pipeline_boxsizer()

        # Task information part
        task_info_staticboxsizer, self.work_folder_text, self.task_name_text = self._init_task_info()
        pipeline_boxsizer.Add(task_info_staticboxsizer, 0, wx.ALL, 10)

        # Notebook for general config, training step, prediction step
        self.nb = wx.Notebook(self._main_panel, -1)

        # General config
        self.general_config_panel = GeneralConfigPanel(self.nb, style=wx.TAB_TRAVERSAL | wx.CLIP_CHILDREN | wx.FULL_REPAINT_ON_RESIZE)
        self.nb.AddPage(self.general_config_panel, 'General config')

        # Training step
        self.train_step_panel = TrainStepPanel(self.nb, style=wx.TAB_TRAVERSAL | wx.CLIP_CHILDREN | wx.FULL_REPAINT_ON_RESIZE)
        self.nb.AddPage(self.train_step_panel, 'Training step')

        # Prediction step
        self.pred_step_panel = PredStepPanel(self.nb, style=wx.TAB_TRAVERSAL | wx.CLIP_CHILDREN | wx.FULL_REPAINT_ON_RESIZE)
        self.nb.AddPage(self.pred_step_panel, 'Prediction step')

        # self.nb.SetSize(self.nb.GetBestSize())
        pipeline_boxsizer.Add(self.nb, 0, wx.ALL, 10)

        # Run task part
        run_task_boxsizer = self._init_run_task_sizer()
        pipeline_boxsizer.Add(run_task_boxsizer, 0, wx.ALL, 10)

        self.boxsizer_main.Add(pipeline_boxsizer, 0, wx.ALL, 10)

        """ Tools """
        tools_boxsizer = self._init_tools_sizer()
        self.boxsizer_main.Add(tools_boxsizer, 0, wx.ALL, 10)

        # Register boxsizer
        self._main_panel.SetSizerAndFit(self.boxsizer_main)
        self.SetClientSize(self._main_panel.GetBestSize())
        self.boxsizer_main.SetSizeHints(self._main_panel)
        self.boxsizer_main.Layout()

    def _create_menubar(self):
        file_menu = wx.Menu()
        about_item = file_menu.Append(-1, '&About...', 'About DeepPhospho desktop app')
        file_menu.AppendSeparator()
        exit_item = file_menu.Append(wx.ID_EXIT)

        menubar = wx.MenuBar()
        menubar.Append(file_menu, '&File')

        self.SetMenuBar(menubar)
        self.Bind(wx.EVT_MENU, self._exit_menu, exit_item)
        self.Bind(wx.EVT_MENU, self._about_menu, about_item)

    def _exit_menu(self, event):
        _ = event
        self.Close(True)

    def _about_menu(self, event):
        _ = event
        wx.MessageBox('About msg', 'About', wx.OK | wx.ICON_INFORMATION)

    def _init_status_bar(self):
        # TODO status bar

        status_bar = self.CreateStatusBar()
        # status_bar.SetStatusText(self._offline_status_bar_dict[self._offline])
        return status_bar

    def _init_desc_text(self):
        static_box = wx.StaticBox(self._main_panel, -1, label='DeepPhospho desktop')
        static_box.SetFont(self._font_boxname)
        static_box_sizer = wx.StaticBoxSizer(static_box, wx.VERTICAL)
        grid_sizer = wx.GridBagSizer(hgap=10, vgap=10)

        desc_text = wx.StaticText(self._main_panel, -1, MainDesc)
        desc_text.SetFont(self._font_static_text)
        grid_sizer.Add(desc_text, pos=(0, 0), span=(1, 1), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTRE_VERTICAL)

        static_box_sizer.Add(grid_sizer, proportion=0, flag=wx.ALIGN_CENTRE_VERTICAL | wx.RIGHT, border=50)
        return static_box_sizer

    def _init_pipeline_boxsizer(self):
        static_box = wx.StaticBox(self._main_panel, -1, label='Pipeline')
        static_box.SetFont(self._font_boxname)
        static_box_sizer = wx.StaticBoxSizer(static_box, wx.VERTICAL)

        _temp_text_parameters = wx.StaticText(self._main_panel, -1, 'Parameters')
        _temp_text_parameters.SetFont(self._font_static_text)
        static_box_sizer.Add(_temp_text_parameters, 0, wx.ALL, 10)

        return static_box_sizer

    def _init_task_info(self):
        static_box = wx.StaticBox(self._main_panel, -1, label='Task information')
        static_box.SetFont(self._font_boxname)
        static_box_sizer = wx.StaticBoxSizer(static_box, wx.VERTICAL)
        grid_sizer = wx.GridBagSizer(hgap=10, vgap=10)

        work_folder_horizon_boxsizer = wx.BoxSizer(wx.HORIZONTAL)
        work_folder_desc_text = wx.StaticText(self._main_panel, -1,
                                              'Work folder (define where to perform DeepPhospho pipeline)',)
        work_folder_desc_text.SetFont(self._font_static_text)
        work_folder_horizon_boxsizer.Add(work_folder_desc_text, 0, wx.ALL, 10)
        select_button = wx.Button(self._main_panel, -1, 'Select')
        select_button.SetFont(self._font_static_text)
        select_button.Bind(wx.EVT_BUTTON, self._event_select_workfolder)
        work_folder_horizon_boxsizer.Add(select_button, 0, wx.ALL, 4)
        grid_sizer.Add(work_folder_horizon_boxsizer, pos=(0, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        work_folder_text = wx.TextCtrl(self._main_panel, -1, size=(900, 30), style=wx.TE_HT_ON_TEXT, name='WorkFolder')
        work_folder_text.SetFont(self._font_search_content)
        grid_sizer.Add(work_folder_text, pos=(1, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)


        default_wf = os.path.join(os.path.abspath('.'), 'demo_work_folder')
        if not os.path.exists(default_wf):
            os.makedirs(default_wf)
        work_folder_text.SetValue(default_wf)

        task_name_desc_boxsizer = wx.BoxSizer(wx.HORIZONTAL)
        task_name_desc_text = wx.StaticText(self._main_panel, -1, 'Task name (the identifier of this task)')
        task_name_desc_text.SetFont(self._font_static_text)
        task_name_desc_boxsizer.Add(task_name_desc_text, 0, wx.ALL, 4)
        grid_sizer.Add(task_name_desc_boxsizer, pos=(2, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        task_name_text = wx.TextCtrl(self._main_panel, -1, PipelineParams['TaskName'], size=(900, 30), style=wx.TE_HT_ON_TEXT, name='TaskName')
        task_name_text.SetFont(self._font_search_content)
        grid_sizer.Add(task_name_text, pos=(3, 0), span=(1, 1), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTRE_VERTICAL)

        static_box_sizer.Add(grid_sizer, proportion=0, flag=wx.ALIGN_CENTRE_VERTICAL | wx.RIGHT, border=50)
        return static_box_sizer, work_folder_text, task_name_text

    def _init_run_task_sizer(self):
        static_box = wx.StaticBox(self._main_panel, -1, label='Run task')
        static_box.SetFont(self._font_boxname)
        static_box_sizer = wx.StaticBoxSizer(static_box, wx.VERTICAL)
        grid_sizer = wx.GridBagSizer(hgap=10, vgap=10)

        run_button = wx.Button(self._main_panel, -1, 'Run', name='run_bottum')
        run_button.SetFont(self._font_static_text)
        run_button.Bind(wx.EVT_BUTTON, self._event_run)  # TODO
        grid_sizer.Add(run_button, pos=(0, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        stop_button = wx.Button(self._main_panel, -1, 'Stop')
        stop_button.SetFont(self._font_static_text)
        stop_button.Bind(wx.EVT_BUTTON, self._event_stop)  # TODO
        grid_sizer.Add(stop_button, pos=(0, 1), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)


        def set_checkbox(e, param, key):
            param[key] = e.GetEventObject().GetValue()

        run_mode_checkbox_train = wx.CheckBox(self, -1, 'Train')
        run_mode_checkbox_train.SetValue(PipelineParams['train'])
        run_mode_checkbox_train.Bind(wx.EVT_CHECKBOX,
                                     lambda e: set_checkbox(e, PipelineParams, 'train'))
        run_mode_checkbox_train.SetFont(self._font_static_text)

        run_mode_checkbox_pred = wx.CheckBox(self, -1, 'Prediction')
        run_mode_checkbox_pred.SetValue(PipelineParams['pred'])
        run_mode_checkbox_pred.Bind(wx.EVT_CHECKBOX,
                                    lambda e: set_checkbox(e, PipelineParams, 'pred'))

        run_mode_checkbox_pred.SetFont(self._font_static_text)

        grid_sizer.Add(run_mode_checkbox_train, pos=(0, 2), span=(1, 1),
                       flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)
        grid_sizer.Add(run_mode_checkbox_pred, pos=(1, 2), span=(1, 1),
                       flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        static_box_sizer.Add(grid_sizer, proportion=0,
                             flag=wx.ALIGN_CENTRE_VERTICAL | wx.RIGHT, border=50)

        return static_box_sizer

    def _init_tools_sizer(self):
        static_box = wx.StaticBox(self._main_panel, -1, label='Tools')
        static_box.SetFont(self._font_boxname)
        static_box_sizer = wx.StaticBoxSizer(static_box, wx.VERTICAL)
        grid_sizer = wx.GridBagSizer(hgap=10, vgap=10)

        build_lib_button = wx.Button(self._main_panel, -1, 'Build library from prediction output')
        build_lib_button.SetFont(self._font_static_text)
        # build_lib_button.Bind(wx.EVT_BUTTON, search_func)  # TODO
        grid_sizer.Add(build_lib_button, pos=(0, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        merge_lib_button = wx.Button(self._main_panel, -1, 'Merge library')
        merge_lib_button.SetFont(self._font_static_text)
        # merge_lib_button.Bind(wx.EVT_BUTTON, search_func)  # TODO
        grid_sizer.Add(merge_lib_button, pos=(0, 1), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        static_box_sizer.Add(grid_sizer, proportion=0, flag=wx.ALIGN_CENTRE_VERTICAL | wx.RIGHT, border=50)
        return static_box_sizer

    def _event_select_workfolder(self, event):
        dlg = wx.DirDialog(self, 'Choose work folder', '.', )
        if dlg.ShowModal() == wx.ID_OK:
            dirname = dlg.GetPath()
        dlg.Destroy()
        self.FindWindowByName('WorkFolder').SetValue(os.path.abspath(dirname))

    def _event_run(self, event):
        self.collect_info_all()
        start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        runner_config = parse_args_from_ui_to_runner(PipelineParams)
        ret = wx.MessageBox(f'Please check the configs below: \n{runner_config}', 'Confirm', wx.OK | wx.CANCEL)
        if ret == wx.OK:
            pass
        else:
            pass
        self.runner_thread = RunnerThread(runner_config, start_time, ui_callback=event)
        self.runner_thread.setDaemon(True)
        self.runner_thread.start()
        event.GetEventObject().Disable()

    def _event_stop(self, event):
        self.runner_thread.terminate()
        # self.runner_thread.join()
        widget = self.FindWindowByName('run_bottum')
        widget.Enable()

    def collect_info_curr_panel(self):
        for name in ['WorkFolder', 'TaskName']:
            widget = self.FindWindowByName(name)
            if widget is not None:
                PipelineParams[name] = widget.GetValue()

    def collect_info_all(self):
        self.collect_info_curr_panel()
        self.general_config_panel.collect_info_curr_panel()
        self.train_step_panel.collect_info_curr_panel()
        self.pred_step_panel.collect_info_curr_panel()


def init_boxsizer_for_ctrltext_with_desc(panel, desc_text, font, ctrl_text_name=None, default_value=None):
    boxsizer = wx.BoxSizer(wx.HORIZONTAL)
    desc_text = wx.StaticText(panel, -1, desc_text)
    desc_text.SetFont(font)
    boxsizer.Add(desc_text, 0, wx.ALL, 10)

    ctrl_text = wx.TextCtrl(panel, -1, default_value, size=(80, 28), style=wx.TE_HT_ON_TEXT, name=ctrl_text_name)
    ctrl_text.SetFont(font)
    boxsizer.Add(ctrl_text, 0, wx.ALL, 10)
    return boxsizer, ctrl_text


class GeneralConfigPanel(wx.Panel):
    def __init__(self, notebook, **kwargs):
        super(GeneralConfigPanel, self).__init__(notebook, **kwargs)

        self._font_boxname = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False)
        self._font_static_text = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False)
        self._font_search_content = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False)
        self._font_listbox = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_LIGHT, False)

        self.boxsizer_main = wx.BoxSizer(wx.VERTICAL)  # Contain common and external tool

        general_config_staticboxsizer, self.rt_pretrain_sub_sizer = self._init_general_config()
        self.boxsizer_main.Add(general_config_staticboxsizer, 0, wx.ALL, 10)

        # Register boxsizer
        self.SetSizerAndFit(self.boxsizer_main)
        self.SetClientSize(self.GetBestSize())
        self.boxsizer_main.SetSizeHints(self)
        self.boxsizer_main.Layout()

    def _init_general_config(self):
        static_box = wx.StaticBox(self, -1, label='General config')
        static_box.SetFont(self._font_boxname)
        static_box_sizer = wx.StaticBoxSizer(static_box, wx.VERTICAL)
        grid_sizer = wx.GridBagSizer(hgap=10, vgap=10)

        param_download_desc_text = wx.StaticText(self, -1, 'To use pre-trained model parameters, please download them here')
        param_download_desc_text.SetFont(self._font_static_text)
        grid_sizer.Add(param_download_desc_text, pos=(0, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        ion_pretrain_boxsizer = wx.BoxSizer(wx.HORIZONTAL)
        ion_pretrain_desc_text = wx.StaticText(self, -1, IonPretrainDesc)
        ion_pretrain_desc_text.SetFont(self._font_static_text)
        ion_pretrain_boxsizer.Add(ion_pretrain_desc_text, 0, wx.ALL, 3)
        ion_pretrain_select_button = wx.Button(self, -1, 'Select')
        ion_pretrain_select_button.SetFont(self._font_static_text)
        ion_pretrain_select_button.Bind(wx.EVT_BUTTON, self._event_select_ion_pretrain)
        ion_pretrain_boxsizer.Add(ion_pretrain_select_button, 0, wx.ALIGN_CENTRE, 3)
        grid_sizer.Add(ion_pretrain_boxsizer, pos=(1, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        ion_pretrain_text = wx.TextCtrl(self, -1, PipelineParams['Pretrain-Ion'], size=(900, 32), style=wx.TE_HT_ON_TEXT, name='Pretrain-Ion')
        ion_pretrain_text.SetFont(self._font_search_content)
        grid_sizer.Add(ion_pretrain_text, pos=(2, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        rt_pretrain_sub_sizer = self._init_rt_pretrain_sizer()
        grid_sizer.Add(rt_pretrain_sub_sizer, pos=(3, 0), span=(1, 1), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTRE_VERTICAL)

        (device_boxsizer, rt_scale_boxsizer,
         max_pep_len_boxsizer,mode_select_boxsizer) = self._init_widgets_below_rt_pretrain()
        grid_sizer.Add(device_boxsizer, pos=(4, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)
        grid_sizer.Add(rt_scale_boxsizer, pos=(5, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)
        grid_sizer.Add(max_pep_len_boxsizer, pos=(6, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)
        grid_sizer.Add(mode_select_boxsizer, pos=(7, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        static_box_sizer.Add(grid_sizer, proportion=0, flag=wx.ALIGN_CENTRE_VERTICAL | wx.RIGHT, border=50)
        return static_box_sizer, rt_pretrain_sub_sizer

    def _init_widgets_below_rt_pretrain(self):
        device_boxsizer = wx.BoxSizer(wx.HORIZONTAL)
        device_desc_text = wx.StaticText(self, -1, 'Device (use which device to run pipeline,\nchoose one from cpu (use cpu only), 0 (use GPU0), 1 (use GPU1), ...')
        device_desc_text.SetFont(self._font_static_text)
        device_boxsizer.Add(device_desc_text, 0, wx.ALL, 10)

        device_text = wx.TextCtrl(self, -1, PipelineParams['Device'], size=(80, 30), style=wx.TE_HT_ON_TEXT, name='Device')
        device_text.SetFont(self._font_static_text)
        device_boxsizer.Add(device_text, 0, wx.ALIGN_CENTRE, 10)

        check_gpu_button = wx.Button(self, -1, 'Check GPUs')
        check_gpu_button.SetFont(self._font_static_text)
        # check_gpu_button.Bind(wx.EVT_BUTTON, search_func)  # TODO
        device_boxsizer.Add(check_gpu_button, 0, wx.ALL, 10)

        rt_scale_boxsizer = wx.BoxSizer(wx.HORIZONTAL)
        rt_scale_desc_text = wx.StaticText(self, -1, '(i)RT scale from')
        rt_scale_desc_text.SetFont(self._font_static_text)
        rt_scale_boxsizer.Add(rt_scale_desc_text, 0, wx.ALL, 10)

        rt_scale_lower_text = wx.TextCtrl(self, -1, PipelineParams['RTScale-lower'], size=(80, 28), style=wx.TE_HT_ON_TEXT, name='RTScale-lower')
        rt_scale_lower_text.SetFont(self._font_static_text)
        rt_scale_boxsizer.Add(rt_scale_lower_text, 0, wx.ALL, 10)

        rt_scale_desc_fillin_text = wx.StaticText(self, -1, 'to')
        rt_scale_desc_fillin_text.SetFont(self._font_static_text)
        rt_scale_boxsizer.Add(rt_scale_desc_fillin_text, 0, wx.ALL, 10)

        rt_scale_upper_text = wx.TextCtrl(self, -1, PipelineParams['RTScale-upper'], size=(80, 28), style=wx.TE_HT_ON_TEXT, name='RTScale-upper')
        rt_scale_upper_text.SetFont(self._font_static_text)
        rt_scale_boxsizer.Add(rt_scale_upper_text, 0, wx.ALL, 10)

        max_pep_len_boxsizer = wx.BoxSizer(wx.HORIZONTAL)
        max_pep_len_desc_text = wx.StaticText(self, -1, 'Max peptide length')
        max_pep_len_desc_text.SetFont(self._font_static_text)
        max_pep_len_boxsizer.Add(max_pep_len_desc_text, 0, wx.ALL, 10)

        max_pep_len_text = wx.TextCtrl(self, -1, PipelineParams['MaxPepLen'], size=(80, 28), style=wx.TE_HT_ON_TEXT, name='MaxPepLen')
        max_pep_len_text.SetFont(self._font_static_text)
        max_pep_len_boxsizer.Add(max_pep_len_text, 0, wx.ALL, 10)

        mode_select_boxsizer = wx.BoxSizer(wx.HORIZONTAL)
        max_pep_len_desc_text = wx.StaticText(self, -1, 'Training mode')
        train_mode_listbox = wx.ListBox(self, -1, choices=['rt', 'ion'], style=wx.LB_SINGLE, name='TrainMode')
        train_mode_listbox.SetFont(self._font_static_text)
        train_mode_listbox.SetSelection(0)
        mode_select_boxsizer.Add(max_pep_len_desc_text, 0, wx.ALL, 10)
        mode_select_boxsizer.Add(train_mode_listbox, 0, wx.ALL, 10)

        return device_boxsizer, rt_scale_boxsizer, max_pep_len_boxsizer,mode_select_boxsizer

    def _event_check_rt_ensemble(self, e):
        PipelineParams['RTEnsemble'] = e.GetEventObject().GetValue()
        self.collect_info_curr_panel()

        self.boxsizer_main.Clear(True)
        general_config_staticboxsizer, self.rt_pretrain_sub_sizer = self._init_general_config()
        self.boxsizer_main.Add(general_config_staticboxsizer, 0, wx.ALL, 10)
        self.boxsizer_main.Layout()

    def _init_rt_pretrain_sizer(self):
        grid_sizer = wx.GridBagSizer(hgap=5, vgap=5)

        rt_pretrain_desc_boxsizer = wx.BoxSizer(wx.HORIZONTAL)
        text = 'Pre-trained RT model (ensemble on)' if PipelineParams['RTEnsemble'] is True else 'Pre-trained RT model (ensemble off)'
        rt_pretrain_desc_text = wx.StaticText(self, -1, text)
        rt_pretrain_desc_text.SetFont(self._font_static_text)
        rt_pretrain_desc_boxsizer.Add(rt_pretrain_desc_text, 0, wx.ALL, 4)

        rt_ensemble_checkbox = wx.CheckBox(self, -1, 'Use RT model ensemble')
        rt_ensemble_checkbox.SetValue(PipelineParams['RTEnsemble'])
        rt_ensemble_checkbox.SetFont(self._font_static_text)
        rt_ensemble_checkbox.Bind(wx.EVT_CHECKBOX, self._event_check_rt_ensemble)
        rt_pretrain_desc_boxsizer.Add(wx.StaticText(self, -1, ' ' * 10), 0, wx.ALL, 4)
        rt_pretrain_desc_boxsizer.Add(rt_ensemble_checkbox, 0, wx.ALL, 4)

        grid_sizer.Add(rt_pretrain_desc_boxsizer, pos=(0, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        if PipelineParams['RTEnsemble'] is True:
            for pos, layer in enumerate([4, 5, 6, 7, 8], 1):
                rt_pretrain_one_row_boxsizer = self._init_rt_pretrain_one_row(layer=layer)
                grid_sizer.Add(rt_pretrain_one_row_boxsizer, pos=(pos, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)
        else:
            rt_pretrain_one_row_boxsizer = self._init_rt_pretrain_one_row(layer=8)
            grid_sizer.Add(rt_pretrain_one_row_boxsizer, pos=(1, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)
        return grid_sizer

    def _init_rt_pretrain_one_row(self, layer):
        rt_pretrain_row_boxsizer = wx.BoxSizer(wx.HORIZONTAL)

        layer_desc_text = wx.StaticText(self, -1, f'Layer {layer}')
        layer_desc_text.SetFont(self._font_static_text)
        rt_pretrain_row_boxsizer.Add(layer_desc_text, 0, wx.ALIGN_CENTRE, 10)

        rt_pretrain_text = wx.TextCtrl(self, -1, PipelineParams[f'Pretrain-RT-{layer}'], size=(800, 28), style=wx.TE_HT_ON_TEXT, name=f'Pretrain-RT-{layer}')
        rt_pretrain_text.SetFont(self._font_search_content)
        rt_pretrain_row_boxsizer.Add(rt_pretrain_text, 0, wx.ALIGN_CENTRE, 10)

        rt_pretrain_select_button = wx.Button(self, -1, 'Select', name=f'Button-Pretrain-RT-{layer}')
        rt_pretrain_select_button.SetFont(self._font_static_text)
        rt_pretrain_select_button.Bind(wx.EVT_BUTTON, self._event_select_rt_pretrain)
        rt_pretrain_row_boxsizer.Add(rt_pretrain_select_button, 0, wx.ALIGN_CENTRE, 10)

        return rt_pretrain_row_boxsizer

    def _event_select_ion_pretrain(self, event):
        dlg = wx.FileDialog(self, 'Choose pre-trained ion intensity model parameter', '.', '', '*', wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetFilename()
        dirname = dlg.GetDirectory()
        dlg.Destroy()
        file_path = os.path.join(dirname, filename)
        self.FindWindowByName(f'Pretrain-Ion').SetValue(file_path)
        PipelineParams[f'Pretrain-Ion'] = file_path

    def _event_select_rt_pretrain(self, event):
        dlg = wx.FileDialog(self, 'Choose pre-trained RT model parameter', '.', '', '*', wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetFilename()
        dirname = dlg.GetDirectory()
        dlg.Destroy()
        button_name = event.GetEventObject().GetName()
        layer = button_name.split('-')[-1]
        file_path = os.path.join(dirname, filename)
        self.FindWindowByName(f'Pretrain-RT-{layer}').SetValue(file_path)
        PipelineParams[f'Pretrain-RT-{layer}'] = file_path

    def collect_info_curr_panel(self):
        for name in ['Pretrain-Ion', *[f'Pretrain-RT-{l}' for l in (4, 5, 6, 7, 8)], 'Device', 'RTScale-lower', 'RTScale-upper', 'MaxPepLen']:
            widget = self.FindWindowByName(name)
            if widget is not None:
                PipelineParams[name] = widget.GetValue()


class TrainStepPanel(wx.Panel):
    def __init__(self, notebook, **kwargs):
        super(TrainStepPanel, self).__init__(notebook, **kwargs)

        self._font_boxname = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False)
        self._font_static_text = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False)
        self._font_search_content = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False)
        self._font_listbox = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_LIGHT, False)

        self.boxsizer_main = wx.BoxSizer(wx.VERTICAL)  # Contain common and external tool

        train_step_staticboxsizer = self._init_train_step()
        self.boxsizer_main.Add(train_step_staticboxsizer, 0, wx.ALL, 10)

        # Register boxsizer
        self.SetSizerAndFit(self.boxsizer_main)
        self.SetClientSize(self.GetBestSize())
        self.boxsizer_main.SetSizeHints(self)
        self.boxsizer_main.Layout()

    def _init_train_step(self):
        static_box = wx.StaticBox(self, -1, label='Training step')
        static_box.SetFont(self._font_boxname)
        static_box_sizer = wx.StaticBoxSizer(static_box, wx.VERTICAL)
        grid_sizer = wx.GridBagSizer(hgap=10, vgap=10)

        train_data_desc_text = wx.StaticText(self, -1, TrainDataDesc)
        train_data_desc_text.SetFont(self._font_static_text)
        grid_sizer.Add(train_data_desc_text, pos=(0, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        train_data_select_button = wx.Button(self, -1, 'Select')
        train_data_select_button.SetFont(self._font_static_text)
        train_data_select_button.Bind(wx.EVT_BUTTON, self._event_select_train_file)
        grid_sizer.Add(train_data_select_button, pos=(0, 1), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        train_data_text = wx.TextCtrl(self, -1, size=(850, 65), style=wx.TE_HT_ON_TEXT | wx.TE_MULTILINE, name='TrainData')
        train_data_text.SetFont(self._font_search_content)
        grid_sizer.Add(train_data_text, pos=(1, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        train_data_format_boxsizer = wx.BoxSizer(wx.HORIZONTAL)
        train_data_format_desc_text = wx.StaticText(self, -1, 'Training data format')
        train_data_format_desc_text.SetFont(self._font_static_text)
        train_data_format_boxsizer.Add(train_data_format_desc_text, 0, wx.ALL, 10)
        format_listbox = wx.ListBox(self, -1, choices=TrainFormatList, style=wx.LB_SINGLE, name='TrainDataFormat')
        format_listbox.SetFont(self._font_static_text)
        format_listbox.SetSelection(0)
        train_data_format_boxsizer.Add(format_listbox, 0, wx.ALL, 10)
        grid_sizer.Add(train_data_format_boxsizer, pos=(2, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        ion_epoch_boxsizer, ion_epoch_ctrltext = init_boxsizer_for_ctrltext_with_desc(
            panel=self, desc_text='Epoch (ion intensity model)', font=self._font_static_text, ctrl_text_name='Epoch-Ion', default_value=PipelineParams['Epoch-Ion'])
        grid_sizer.Add(ion_epoch_boxsizer, pos=(3, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        rt_epoch_boxsizer, rt_epoch_ctrltext = init_boxsizer_for_ctrltext_with_desc(
            panel=self, desc_text='Epoch (RT model)', font=self._font_static_text, ctrl_text_name='Epoch-RT', default_value=PipelineParams['Epoch-RT'])
        grid_sizer.Add(rt_epoch_boxsizer, pos=(4, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        ion_bs_boxsizer, ion_bs_ctrltext = init_boxsizer_for_ctrltext_with_desc(
            panel=self, desc_text='Batch size (ion intensity model)', font=self._font_static_text,
            ctrl_text_name='BatchSize-Ion', default_value=PipelineParams['BatchSize-Ion'])
        grid_sizer.Add(ion_bs_boxsizer, pos=(5, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        rt_bs_boxsizer, rt_bs_ctrltext = init_boxsizer_for_ctrltext_with_desc(
            panel=self, desc_text='Batch size (RT model)', font=self._font_static_text,
            ctrl_text_name='BatchSize-RT', default_value=PipelineParams['BatchSize-RT'])
        grid_sizer.Add(rt_bs_boxsizer, pos=(6, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        init_lr_boxsizer, init_lr_ctrltext = init_boxsizer_for_ctrltext_with_desc(
            panel=self, desc_text='Initial learning rate', font=self._font_static_text,
            ctrl_text_name='InitLR', default_value=PipelineParams['InitLR'])
        grid_sizer.Add(init_lr_boxsizer, pos=(7, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        static_box_sizer.Add(grid_sizer, proportion=0, flag=wx.ALIGN_CENTRE_VERTICAL | wx.RIGHT, border=50)
        return static_box_sizer

    def _event_select_train_file(self, event):
        dlg = wx.FileDialog(self, 'Choose a file', '.', '', '*', wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetFilename()
        dirname = dlg.GetDirectory()
        dlg.Destroy()
        self.FindWindowByName(f'TrainData').SetValue(os.path.join(dirname, filename))

    def collect_info_curr_panel(self):
        for name in ['TrainData', 'TrainDataFormat', 'TrainMode', 'Epoch-Ion', 'Epoch-RT', 'BatchSize-Ion', 'BatchSize-RT', 'InitLR']:
            widget = self.FindWindowByName(name)
            if widget is not None:
                if name == 'TrainDataFormat':
                    PipelineParams[name] = TrainFormatList[widget.GetSelection()]
                elif name == 'TrainMode':

                    PipelineParams[name] = ['rt', 'ion'][widget.GetSelection()]
                else:
                    PipelineParams[name] = widget.GetValue()


class PredStepPanel(wx.Panel):
    def __init__(self, notebook, **kwargs):
        super(PredStepPanel, self).__init__(notebook, **kwargs)

        # Status
        self.pred_input_widgets = {}  # with format {'': (pred_format_widget, pred_file_text_widget)}
        self.pred_input_widgets_num = 0

        # Font
        self._font_boxname = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False)
        self._font_static_text = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False)
        self._font_search_content = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False)
        self._font_listbox = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_LIGHT, False)

        self.boxsizer_main = wx.BoxSizer(wx.VERTICAL)  # Contain common and external tool

        pred_step_staticboxsizer, self.pred_step_grid_sizer = self._init_pred_step()
        self.boxsizer_main.Add(pred_step_staticboxsizer, 0, wx.ALL, 10)

        # Register boxsizer
        self.SetSizerAndFit(self.boxsizer_main)
        self.SetClientSize(self.GetBestSize())
        self.boxsizer_main.SetSizeHints(self)
        self.boxsizer_main.Layout()

    def _init_pred_step(self):
        static_box = wx.StaticBox(self, -1, label='Prediction step')
        static_box.SetFont(self._font_boxname)
        static_box_sizer = wx.StaticBoxSizer(static_box, wx.VERTICAL)
        grid_sizer = wx.GridBagSizer(hgap=10, vgap=10)

        pred_input_desc_text = wx.StaticText(self, -1, PredInputDesc)
        pred_input_desc_text.SetFont(self._font_static_text)
        grid_sizer.Add(pred_input_desc_text, pos=(0, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        add_row_button = wx.Button(self, -1, 'Add row')
        add_row_button.SetFont(self._font_static_text)
        add_row_button.Bind(wx.EVT_BUTTON, self._event_add_row)
        grid_sizer.Add(add_row_button, pos=(1, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        pred_row_boxsizer = self._init_one_pred_row()
        self.pred_input_widgets_num += 1
        grid_sizer.Add(pred_row_boxsizer, pos=(2, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)

        static_box_sizer.Add(grid_sizer, proportion=0, flag=wx.ALIGN_CENTRE_VERTICAL | wx.RIGHT, border=50)
        return static_box_sizer, grid_sizer

    def _event_add_row(self, event):
        pred_row_boxsizer = self._init_one_pred_row()
        self.pred_step_grid_sizer.Add(pred_row_boxsizer, pos=(2 + self.pred_input_widgets_num, 0), span=(1, 1), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTRE_VERTICAL)
        self.pred_input_widgets_num += 1
        self.boxsizer_main.Layout()

    def _init_one_pred_row(self):
        boxsizer = wx.BoxSizer(wx.HORIZONTAL)

        pred_format_choice = wx.Choice(self, -1, choices=PredictionFormatList, name=f'Choice-{self.pred_input_widgets_num}')
        pred_format_choice.SetFont(self._font_static_text)
        pred_format_choice.SetSelection(0)
        boxsizer.Add(pred_format_choice, 0, wx.ALL, 10)

        pred_file_text = wx.TextCtrl(self, -1, size=(900, 30), style=wx.TE_HT_ON_TEXT, name=f'CtrlText-{self.pred_input_widgets_num}')
        pred_file_text.SetFont(self._font_search_content)
        boxsizer.Add(pred_file_text, 0, wx.ALL, 10)

        pred_file_select_button = wx.Button(self, -1, 'Select', name=f'Button-{self.pred_input_widgets_num}')
        pred_file_select_button.SetFont(self._font_static_text)
        pred_file_select_button.Bind(wx.EVT_BUTTON, self._event_select_pred_file)
        boxsizer.Add(pred_file_select_button, 0, wx.ALL, 10)

        return boxsizer

    def _event_select_pred_file(self, event):
        dlg = wx.FileDialog(self, 'Choose a file', '.', '', '*', wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetFilename()
        dirname = dlg.GetDirectory()
        dlg.Destroy()
        button_name = event.GetEventObject().GetName()
        idx = button_name.split('-')[-1]
        self.FindWindowByName(f'CtrlText-{idx}').SetValue(os.path.join(dirname, filename))

    def collect_info_curr_panel(self):
        PipelineParams['PredInput'] = []
        PipelineParams['PredInputFormat'] = []

        for idx in range(self.pred_input_widgets_num):
            choice = PredictionFormatList[self.FindWindowByName(f'Choice-{idx}').GetSelection()]
            ctrl_text = self.FindWindowByName(f'CtrlText-{idx}').GetValue()
            if ctrl_text is not None and ctrl_text != '':
                # TODO check data path
                PipelineParams['PredInput'].append(ctrl_text)
                PipelineParams['PredInputFormat'].append(choice)
