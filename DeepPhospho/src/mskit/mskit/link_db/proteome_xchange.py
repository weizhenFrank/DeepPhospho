import requests
import pandas as pd
from ftplib import FTP

from mskit.constants import ProteomeXchange as PXConst


class PrideFTPDownloader(object):
    def __init__(self, project_id=None):
        self._id = project_id

        self.FileListUrl = None
        self.filelist = None

        self.fileinfo_dict = None
        self.fileinfo_df = None

        self.excluded_files = None
        self._ftp = FTP()

    def get_project_id(self):
        return self._id

    def set_project_id(self, project_id):
        self._id = project_id

    project_id = property(get_project_id, set_project_id, doc='''
    Project ID of ProteomeXchange, like PXD004732''')

    def set_ftp_debuglevel(self, level):
        self._ftp.set_debuglevel(level)

    def get_file_info(self):
        self.FileListUrl = PXConst.Pride.ListQueryUrl + self._id
        request = requests.get(self.FileListUrl)
        if request.status_code != 200:
            raise requests.exceptions.ConnectionError(request.status_code)
        request_json = request.json()
        raw_filelist = request_json['list']
        self.fileinfo_dict = {_['fileName']: _ for _ in raw_filelist}
        self.fileinfo_df = pd.DataFrame(raw_filelist)
        self.filelist = self.fileinfo_df['fileName'].tolist()
        return self.fileinfo_dict

    def get_filelist(self):
        return self.filelist

    def get_exclusion(self):
        return self.excluded_files

    def set_exclusion(self, exclusion):
        self.excluded_files = exclusion

    exclusion = property(get_exclusion, set_exclusion, doc='''''')

    def link_ftp(self):
        try:
            welcome_message = self._ftp.connect(PXConst.Pride.FTP_IP)
        except TimeoutError:
            raise TimeoutError(f'Time out when connect to {PXConst.Pride.FTP_IP}')
        else:
            if '220' not in welcome_message or welcome_message != '220-Welcome to ftp.ebi.ac.uk\n220 ':
                print(f'Connected to ftp server but no correct connect code: {welcome_message}')

    def login_ftp(self, email=None, user='anonymous', password=None):
        if not password:
            if email:
                password = email
            else:
                password = 'your@email.address'
        try:
            login_message = self._ftp.login(user, password)
        except Exception:
            print(f'Login error when login with user=\'{user}\', password=\'{password}\'')
            raise
        else:
            if '230' not in login_message or login_message != '230 Login successful.':
                print(f'Login but no correct login code')

    def ftp_cwd(self):
        pass
    '250 Directory successfully changed.'

    def download(self, ):
        pass

    def __call__(self, filename):
        if self.fileinfo_dict:
            if isinstance(filename, str):
                try:
                    return self.fileinfo_dict[filename]
                except KeyError:
                    raise KeyError(f'No such file {filename}')
            else:
                raise TypeError(f'File name illegal {filename}')
        else:
            raise

    def __del__(self):
        try:
            self._ftp.quit()
        except AttributeError:
            self._ftp.close()

    @staticmethod
    def asp_cmd(year, month, project_id, file_name, target_address):
        ascp_cmd = ' '.join([r'ascp',
                             r'-QT -l 300m -P33001 -i',
                             r'asperaweb_id_dsa.openssh',
                             r'prd_ascp@fasp.ebi.ac.uk:pride/data/archive/{year}/{month}/{project_id}/{file_name}'.format(
                                 year=year, month=month, project_id=project_id, file_name=file_name),
                             rf'{target_address}',
                             ])
