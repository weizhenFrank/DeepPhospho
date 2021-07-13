

MainDesc = ('Main Description'
            '')

IonPretrainDesc = 'Pre-trained ion intensity model'

'(if defined, model will be fine-tuned in training step or\n'
'directly used to predict fragment intensity for peptides in prediction input)'

TrainDataDesc = 'Training data (use which data to train new models or fine-tune pre-trained models)'
TrainFormatList = ['SNLib', 'MQ1.5', 'MQ1.6']

PredInputDesc = 'Prediction input (for the description of prediction input format, please have a look at our GitHub repository)'
PredictionFormatList = ['SNLib', 'SNResult', 'MQ1.5', 'MQ1.6', 'PepSN13', 'PepMQ1.5', 'PepMQ1.6', 'PepComet', 'PepDP']
