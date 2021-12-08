

REPO = 'https://github.com/weizhenFrank/DeepPhospho'

MainDesc = ('DeepPhospho desktop provides the full supports for DeepPhospho runner with graphical user interface, '
            'to make the running of DeepPhospho pipeline easily.\n'
            'This app integrates functions of training, prediction, and library generation.\n'
            f'Please have a look at GitHub page {REPO} for details of usages of desktop app and command line tools.')

IonPretrainDesc = 'Pre-trained ion intensity model'

'(if defined, model will be fine-tuned in training step or\n'
'directly used to predict fragment intensity for peptides in prediction input)'

DeviceDesc = ('Device (use which device to run pipeline.\n'
              'Choose one from cpu (use cpu only), 0 (use GPU0), 1 (use GPU1), ...')

TrainDataDesc = 'Training data (use which data to train new models or fine-tune pre-trained models)'
TrainFormatList = ['SNLib', 'MQ1.5', 'MQ1.6']

PredInputDesc = 'Prediction input\n(for the description of prediction input format, please have a look at our GitHub repository)'
PredictionFormatList = ['SNLib', 'SNResult', 'MQ1.5', 'MQ1.6', 'PepSN13', 'PepMQ1.5', 'PepMQ1.6', 'PepComet', 'PepDP']

BuildLibDesc = ('  Library building with DeepPhospho raw predictions.\n'
                '  This tool needs two input files to build a spectral library in tab-separate flatten text format (Spectronaut and DIA-NN compatible).\n'
                '  Two input files are prediction results from DeepPhospho ion intensity model and retention time model.\n'
                '    - Fragment ion intensity result: after prediction done, there will be a JSON file stored in work folder of the certain distance. '
                'The file name is usually the combination of distance name and "-PredOutput.json".\n'
                '    - Retention time result: after prediction done, a file named "Prediction.txt" will be created in the result folder.')

MergeLibDesc = ('  Merge several libraries to one.\n'
                '  This tool needs at least two library files as input and output a merged non-redundent spectral library.\n'
                '  Different with the generally used consensus library generation method, '
                'this tool will use the first library as the main library, '
                'which means all precursors in this library will be retained, '
                'and the same precursors in following libraries will be directly deleted.\n'
                '  This method might not suitable for experimental libraries, '
                'but it fits the need of current case, '
                'since all input libraries are from DeepPhospho prediction (from the same source).')
