# DeepPhospho: impoves spectra and RT prediction for phosphopeptides

## Contents

* [installation](#Installation)
* [1.1 DeepPhospho installation](#11-deepphospho-installation)
    * [1.2 conda environment installation](#12-conda-environment-installation)
    
* [2. Train model](#2-train-model)
   * [2.1 Train model on RT task](#21-train-model-on-rt-task)
   * [2.2 Train model on Ion intensity task](#22-train-model-on-ion-intensity-task)

* [3. Predict ](#3-predict)
   * [3.1 predict the RT](#31-predict-the-rt)
   * [3.2 predict the Ion intensity](#32-predict-the-ion-intensity)
   
# Setup DeepPhospho

## Get DeepPhospho

* Clone this repository

  ```
  git clone https://github.com/weizhenFrank/DeepPhospho.git
  ```

* Or install this package from PyPi

  ```
  pip install xxx
  ```

## Conda environment

```
conda env create -f DeepPhospho_ENV.yml
```

# 2. Train model

## 2.1 Train model on RT task
* Go to the file 
> PutOnGitHub/IonInten/configs/config_main.py 
* change the 
`mode = 'RT'`
* chaneg the `GPU_INDEX` according to your machine in `TRAINING_HYPER_PARAM`
* change the  `data_name = "U2OS_DIA"` (one of  `"U2OS_DIA_RT"` `"PhosDIA_DIA18_finetune_RT"` `"PhosDIA_DDA_RT"`)
* (optional) use the checkpoint in checkpoints folder, and change  `pretrain_param='<checkpoint path>'` in `RT_TRAINING_HYPER_PARAM`
* run the main.py
`python main.py`
* the output will be saved in the `result/RT`

## 2.2 Train model on Ion intensity task
* Go to the file 
> PutOnGitHub/IonInten/configs/config_main.py 
* change the 
`mode = 'IonIntensity'`
* chaneg the `GPU_INDEX` according to your machine in  `TRAINING_HYPER_PARAM`
* change the `data_name = "U2OS_DIA"` (one of  `"U2OS_DIA"` `"PhosDIA_DIA18"`  `"PhosDIA_DDA"`)
* (optional) use the checkpoint in checkpoints folder, and change `pretrain_param='<checkpoint path>'` in `Intensity_TRAINING_HYPER_PARAM`
* run the main_ion.py
`python main_ion.py`
* the output will be saved in the `result/IonInten/`

# 3. Predict 
## 3.1 predict the RT
* Go to the file
> PutOnGitHub/IonInten/configs/config_main.py 
* change the 
`"model_name": "LSTMTransformerEnsemble",`
* Go to the file
> visualization.py
* use the checkpoint in checkpoints folder, and change `model_arch_path={}`. The key of the dict is the number of encoder layer, the corresponding value is the path of the model's checkpoint.
* run visualization.py
`python visualization.py`
* the prediction will be saved in the `result/RT/Analysis`

## 3.2 predict the Ion intensity
* Go to the file
> PutOnGitHub/IonInten/configs/config_main.py 
* change the 
`"model_name": "LSTMTransformer",`
* Go to the file
> visualization_Ion.py
* use the checkpoint in checkpoints folder, and change `load_model_path='<checkpoint path>'`. 
* run visualization_Ion.py
`python visualization_Ion.py`
* the prediction will be saved in the `result/IonInten/Analysis`
