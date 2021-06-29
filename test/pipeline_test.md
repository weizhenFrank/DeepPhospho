* Test DeepPhospho pipeline

## General usage with fine-tuning and prediction

```shell
python ./run_deep_phospho.py -w ./test/TestPipeline-GeneralUsage -t TestGeneralUsage -tf ./test/PipelineTestData/SNLib-ForTraining.xls -tt SNLib -pf ./test/PipelineTestData/PredInput-Format_MQ1.6.txt -pt MQ1.6 -pf ./test/PipelineTestData/PredInput-Format_PepDP.txt -pt PepDP -d 0 -ie 2 -re 2 -ibs 128 -rbs 128 -lr 0.0001 -ml 74 -rs *-100,200 -en -no_time -m
```

## Partial fine-tuning

### (pass) Existed ion model and fine-tune ensemble RT

```shell
python ./run_deep_phospho.py -w ./test/TestPipeline-ExistIon_FineTuneRTEnsemble -t Test-ExistIon_FineTuneRTEnsemble -tf ./test/PipelineTestData/SNLib-ForTraining.xls -tt SNLib -pf ./test/PipelineTestData/PredInput-Format_MQ1.6.txt -pt MQ1.6 -pf ./test/PipelineTestData/PredInput-Format_PepDP.txt -pt PepDP -d 0 -ie 2 -re 2 -ibs 128 -rbs 128 -lr 0.0001 -ml 74 -rs *-100,200 -en -no_time -m -ion_model ./test/TestPipeline-GeneralUsage/TestGeneralUsage-IonModel/ckpts/best_model.pth
```

### (pass) Existed ion model and fine-tune ensemble RT with 3 existed RT models

```shell
python ./run_deep_phospho.py -w ./test/TestPipeline-ExistIon_ExistRT578_FineTuneRTEnsemble -t Test-ExistIon_ExistRT578_FineTuneRTEnsemble -tf ./test/PipelineTestData/SNLib-ForTraining.xls -tt SNLib -pf ./test/PipelineTestData/PredInput-Format_MQ1.6.txt -pt MQ1.6 -pf ./test/PipelineTestData/PredInput-Format_PepDP.txt -pt PepDP -d 0 -ie 2 -re 2 -ibs 128 -rbs 128 -lr 0.0001 -ml 74 -rs *-100,200 -en -no_time -m -ion_model ./test/TestPipeline-GeneralUsage/TestGeneralUsage-IonModel/ckpts/best_model.pth -rt_model_5 ./test/TestPipeline-GeneralUsage/TestGeneralUsage-RTModel-5/ckpts/best_model.pth -rt_model_7 ./test/TestPipeline-GeneralUsage/TestGeneralUsage-RTModel-7/ckpts/best_model.pth -rt_model_8 ./test/TestPipeline-GeneralUsage/TestGeneralUsage-RTModel-8/ckpts/best_model.pth
```

### (pass) Existed ion model and fine-tune single RT

```shell
python ./run_deep_phospho.py -w ./test/TestPipeline-ExistIon_FineTuneRTSingle -t Test-ExistIon_FineTuneRTSingle -tf ./test/PipelineTestData/SNLib-ForTraining.xls -tt SNLib -pf ./test/PipelineTestData/PredInput-Format_MQ1.6.txt -pt MQ1.6 -pf ./test/PipelineTestData/PredInput-Format_PepDP.txt -pt PepDP -d 0 -ie 2 -re 2 -ibs 128 -rbs 128 -lr 0.0001 -ml 74 -rs *-100,200 -no_time -m -ion_model ./test/TestPipeline-GeneralUsage/TestGeneralUsage-IonModel/ckpts/best_model.pth
```

### (error) Existed ion model but parameter file not found

```shell
python ./run_deep_phospho.py -w ./test/TestPipeline-ExistIon_FilePathError -t Test-ExistIon_FilePathError -tf ./test/PipelineTestData/SNLib-ForTraining.xls -tt SNLib -pf ./test/PipelineTestData/PredInput-Format_MQ1.6.txt -pt MQ1.6 -pf ./test/PipelineTestData/PredInput-Format_PepDP.txt -pt PepDP -d 0 -ie 2 -re 2 -ibs 128 -rbs 128 -lr 0.0001 -ml 74 -rs *-100,200 -en -no_time -m -ion_model ./test/TestPipeline-GeneralUsage/TestGeneralUsage-IonModel/ckpts/best_model.pth-Error
```

## No training data

### (pass) existed ion model and RT model 8 with no ensemble (training data also passed)

```shell
python ./run_deep_phospho.py -w ./test/TestPipeline-ExistIon_RT8_NoEnsemble_PredOnly_TrainPassed -t Test-ExistIon_RT8_NoEnsemble_PredOnly_TrainPassed -tf ./test/PipelineTestData/SNLib-ForTraining.xls -tt SNLib -pf ./test/PipelineTestData/PredInput-Format_MQ1.6.txt -pt MQ1.6 -pf ./test/PipelineTestData/PredInput-Format_PepDP.txt -pt PepDP -d 0 -ie 2 -re 2 -ibs 128 -rbs 128 -lr 0.0001 -ml 74 -rs *-100,200 -no_time -m -ion_model ./test/TestPipeline-GeneralUsage/TestGeneralUsage-IonModel/ckpts/best_model.pth -rt_model_8 ./test/TestPipeline-GeneralUsage/TestGeneralUsage-RTModel-8/ckpts/best_model.pth
```

### (pass) existed ion model and RT model 8 with no ensemble (no training data)

```shell
python ./run_deep_phospho.py -w ./test/TestPipeline-ExistIon_RT8_NoEnsemble_PredOnly -t Test-ExistIon_RT8_NoEnsemble_PredOnly -pf ./test/PipelineTestData/PredInput-Format_MQ1.6.txt -pt MQ1.6 -pf ./test/PipelineTestData/PredInput-Format_PepDP.txt -pt PepDP -d 0 -ie 2 -re 2 -ibs 128 -rbs 128 -lr 0.0001 -ml 74 -rs *-100,200 -no_time -m -ion_model ./test/TestPipeline-GeneralUsage/TestGeneralUsage-IonModel/ckpts/best_model.pth -rt_model_8 ./test/TestPipeline-GeneralUsage/TestGeneralUsage-RTModel-8/ckpts/best_model.pth
```

### (pass) existed ion model and all 5 RT models with ensemble

```shell
python ./run_deep_phospho.py -w ./test/TestPipeline-ExistIon_AllRT_Ensemble_PredOnly -t Test-ExistIon_AllRT_Ensemble_PredOnly -pf ./test/PipelineTestData/PredInput-Format_MQ1.6.txt -pt MQ1.6 -pf ./test/PipelineTestData/PredInput-Format_PepDP.txt -pt PepDP -d 0 -ie 2 -re 2 -ibs 128 -rbs 128 -lr 0.0001 -ml 74 -rs *-100,200 -en -no_time -m -ion_model ./test/TestPipeline-GeneralUsage/TestGeneralUsage-IonModel/ckpts/best_model.pth -rt_model_4 ./test/TestPipeline-GeneralUsage/TestGeneralUsage-RTModel-4/ckpts/best_model.pth -rt_model_5 ./test/TestPipeline-GeneralUsage/TestGeneralUsage-RTModel-5/ckpts/best_model.pth -rt_model_6 ./test/TestPipeline-GeneralUsage/TestGeneralUsage-RTModel-6/ckpts/best_model.pth -rt_model_7 ./test/TestPipeline-GeneralUsage/TestGeneralUsage-RTModel-7/ckpts/best_model.pth -rt_model_8 ./test/TestPipeline-GeneralUsage/TestGeneralUsage-RTModel-8/ckpts/best_model.pth
```

### (error) existed ion model and existed RT models 4 and 5 out of [4, 5, 6, 7, 8] with ensemble

```shell
python ./run_deep_phospho.py -w ./test/TestPipeline-ExistIon_RT45_Ensemble_PredOnly_Error -t Test-ExistIon_RT45_Ensemble_PredOnly_Error -pf ./test/PipelineTestData/PredInput-Format_MQ1.6.txt -pt MQ1.6 -pf ./test/PipelineTestData/PredInput-Format_PepDP.txt -pt PepDP -d 0 -ie 2 -re 2 -ibs 128 -rbs 128 -lr 0.0001 -ml 74 -rs *-100,200 -en -no_time -m -ion_model ./test/TestPipeline-GeneralUsage/TestGeneralUsage-IonModel/ckpts/best_model.pth -rt_model_4 ./test/TestPipeline-GeneralUsage/TestGeneralUsage-RTModel-4/ckpts/best_model.pth -rt_model_5 ./test/TestPipeline-GeneralUsage/TestGeneralUsage-RTModel-5/ckpts/best_model.pth
```

### (error) no existed models

```shell
python ./run_deep_phospho.py -w ./test/TestPipeline-NoExistedModel_PredOnly_Error -t Test-NoExistedModel_PredOnly_Error -pf ./test/PipelineTestData/PredInput-Format_MQ1.6.txt -pt MQ1.6 -pf ./test/PipelineTestData/PredInput-Format_PepDP.txt -pt PepDP -d 0 -ie 2 -re 2 -ibs 128 -rbs 128 -lr 0.0001 -ml 74 -rs *-100,200 -en -no_time -m
```





