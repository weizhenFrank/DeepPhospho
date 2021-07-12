

MainDesc = ('Main Description\n'
            'https://github.com/weizhenFrank/DeepPhospho')

IonPretrainDesc = 'Pre-trained ion intensity model'

'(if defined, model will be fine-tuned in training step or\n'
'directly used to predict fragment intensity for peptides in prediction input)'

TrainDataDesc = 'Training data (use which data to train new models or fine-tune pre-trained models)'
TrainFormatList = ['SNLib', 'MQ1.5', 'MQ1.6']

PredInputDesc = 'Prediction input\n(for the description of prediction input format, please have a look at our GitHub repository)'
PredictionFormatList = ['SNLib', 'SNResult', 'MQ1.5', 'MQ1.6', 'PepSN13', 'PepMQ1.5', 'PepMQ1.6', 'PepComet', 'PepDP']

BuildLibDesc = ('  Library building with DeepPhospho raw predictions.\n'
                '  This tool needs two input files to build a spectral library in tab-separate flatten text format (Spectronaut and DIA-NN compatible).\n'
                '  Two input files are prediction results from DeepPhospho ion intensity model and retention time model.\n'
                '    - Fragment ion intensity result: after prediction done, there will be a JSON file stored in work folder of the certain distance. '
                'The file name is usually the combination of distance name and -PredOutput.json\n'
                '    - Retention time result: after prediction done, a file named Prediction.txt will be created in the result folder.')
