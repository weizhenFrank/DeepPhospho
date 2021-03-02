# DeepPhospho: impoves spectral library generation for DIA phosphoproteomics

## Contents

* [Setup DeepPhospho](#setup)
   * [Get DeepPhospho](#get)
   * [Setup conda environment](#conda_env)
   * [Use GPU (optional)](#use_gpu)
* [Quick start](#start)
    * [Predict your data with default params](#start_pred)
* [DeepPhospho configs](#configs)
    * [Configs for ion intensity model](#ion_config)
    * [Configs for RT model](#rt_config)
* [Predict spec and iRT with DeepPhospho](#predict)
    * [Predict spec ion intensity](#pred_spec)
    * [Predict peptide iRT](#pred_rt)
* [Training and fine-tuning model params](#train)
   * [Use the data used in DeepPhospho](#deepphospho_data)
   * [Prepare your customized training data](#prepare_cust_data)
   * [Train ion intensity model with or without our pre-trained params](#train_ion)
   * [Train RT model with or without our pre-trained params](#train_rt)
* [From initial DIA library to DeepPhospho improved library](#lib)
   * [Recommendations for initial library preparation](#lib_recommend)
   * [Check the library and convert it to DeepPhospho specific input formats](#convert_lib_format)
   * [Fine-tuning the models with the customized data](#finetune_with_cust_data)
   * [Generate a DeepPhospho improved library](#gen_lib)
   * [Other details](#other_details)
* [License](#license)
* [Publication](#pub)

# <span id="setup">Setup DeepPhospho</span>

## <span id="get">Get DeepPhospho</span>

* Clone this repository with git

  ```
  git clone https://github.com/weizhenFrank/DeepPhospho.git
  ```

* Or download the repository directly on github by opening

  ```
  https://github.com/weizhenFrank/DeepPhospho
  ```

## <span id="conda_env">Setup conda environment</span>

* DeepPhospho relies on PyTorch and some common packages like numpy

* We recommend conda to manage the enviroment, and the lastest version of Anaconda can be downloaded here: [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)

* In this repository, we supported a conda enviroment config file for quick setup

  ```
  conda env create -f DeepPhospho_ENV.yaml -n deep_phospho
  ```

## <span id="use_gpu">Use GPU (optional)</span>

* DeepPhospho can be runned with GPU or with only CPU
* To use GPU, please confirm the information of your device and install suitable drive ([CUDA](https://developer.nvidia.com/cuda-downloads) (10.0 or 10.1 is recommended if you use the conda enviroment provided in this repository) and [cuDNN](https://developer.nvidia.com/cudnn))

# <span id="start">Quick start</span>

## <span id="start_pred">Predict your data with default params</span>

* Only an input file is needed if you would like to use the default settings and we provided model parameters
* Here, we will use our demo input file as an example, and you can create your own input by your custom script or [using our script](#create_input) to convert files from MaxQuant (1.5- and 1.6+), Spectronaut (13+), and Comet
* Below is a 3 step example for ion intensity prediction:
  1. Change the directory to DeepPhospho main folder
  2. Check config_ion_model.py
     * check WorkFolder is 'Here' or set any other place you want
     * check TaskPurpose is 'Predict'
     * make sure you have downloaded the model parameter and unzipped them in the folder PretrainParams
     * check PretrainParam is correctly pointed to one param file
     * check PredInputPATH in Intensity_DATA_CFG is "./demo/IonInput.txt" or set to other input you want
  3. Run `python pred_ion.py`(please also make sure the config_path in line 37 of pred_ion.py is an empty string)
* RT prediction is same as ion prediction, but we use ensembl to make more accurate prediction. The parameters for prediction is not PretrainParam but ParamsForPred, and it is consisted of the key-value pairs of decoder layer: param path.

# <span id="configs">DeepPhospho configs</span>

* We provide a flexible config import method

* For both ion intensity model and RT model, there are three ways to specify the config file:

  1. directly change the config file 'config_ion_model.py' or 'config_rt_model.py' and run scripts with no further changes

  2. fill in config template files in json format (stored in DeepPhospho main folder) and fill 'config_path' in train or pred script

  3. fill in config template files in json format and pass it as an argument

     ```shell
     python train_ion.py path/to/your/config.json
     ```

* For convenient use, we also provided an argument parser

  * `-c path/to/your/config.json` will force the config file to be set as the file you provided in command line
  * `-g [int] or -g cpu` will overwrite the GPU_INDEX in config, this will be useful to start multi tasks on different device in one time
  * `-l [int]` will overwrite the num_encd_layer in config, which indicates the encoder layer number of transformer
  * -e and -d will overwrite experiment name and dataset name, respectively

* For more information, run `python [any train or pred script] --help`

## <span id="ion_config">Configs for ion intensity model</span>



## <span id="rt_config">Configs for RT model</span>







# <span id="predict">Predict spec and RT with DeepPhospho</span>

## <span id="create_input">Create your own prediction input</span>

## <span id="pred_spec">Predict spec ion intensity</span>

## <span id="pred_rt">Predict peptide iRT</span>

# <span id="train">Training and fine-tuning model params</span>

## <span id="deepphospho_data">Use the data used in DeepPhospho</span>

## <span id="prepare_cust_data">Prepare your customized training data</span>

## <span id="train_ion">Train ion intensity model with or without our pre-trained params</span>

## <span id="train_rt">Train RT model with or without our pre-trained params</span>

# <span id="lib">From initial DIA library to DeepPhospho improved library</span>

## <span id="lib_recommend">Recommendations for initial library preparation</span>

* Though the initial data is called library here, it doesn't indicates the spectral library only. Instead, DeepPhospho workflow makes the spectral library generation much freer.

* All data source that includes the fragment intensity and peptide RT/iRT is satisfied, including but not limited to the spectral library from Spectranaut, OpenSwath and others, and raw data search results from MaxQuant, Comet and others.

* Here we provided [the scripts]() to convert Spectranaut library (exported as .xls) and MaxQuant search result (msms.txt in result folder) to DeepPhospho input formats.

  ```python
  
  ```

* In previous study, the input data is always under some filtration to get much higher quality, but in this work, we didn't see obvious increase on evaluation metrics with different filtration conditions. So we didn't provide the data filtration in scripts for format converting.

## <span id="convert_lib_format">Check the library and convert it to DeepPhospho specific input formats</span>

## <span id="finetune_with_cust_data">Fine-tuning the models with the customized data</span>

## <span id="gen_lib">Generate a DeepPhospho improved library</span>



## <span id="other_details">Other details</span>

* In this document, we described the DeepPhospho usages as detailed as possible, and this document only contains model and library generation information as a supplement for our paper.
* We also discussed many details in the method part in our paper, please have a look at [here](). The methods in paper involves:
  * 



# <span id="license">License</span>

# <span id="pub">Publication</span>











* [2. Train model](#2-train-model)
  * [2.1 Train model on RT task](#21-train-model-on-rt-task)
  * [2.2 Train model on Ion intensity task](#22-train-model-on-ion-intensity-task)
* [3. Predict ](#3-predict)
  * [3.1 predict the RT](#31-predict-the-rt)
  * [3.2 predict the Ion intensity](#32-predict-the-ion-intensity)



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

* 







