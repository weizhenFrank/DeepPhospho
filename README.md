# DeepPhospho: impoves spectral library generation for DIA phosphoproteomics

## Contents

* [Setup DeepPhospho](#setup)
   * [Get DeepPhospho](#get)
   * [Setup conda environment](#conda_env)
   * [Download pre-trained model parameters](#down_pretain)
   * [Use GPU (optional)](#use_gpu)
* [Quick start (from library/result file to DeepPhospho generated library in one command)](#start)
    * [Command template](#command)
    * [Introduction to arguments](#argu_intro)
* [DeepPhospho configs](#configs)
    * [Configs for ion intensity model](#ion_config)
    * [Configs for RT model](#rt_config)
    * [Use config file and overwrite values with command line arguments](#import_config)
* [Train and predict manurally](#manu)
   * [Start demo for model training](#train_demo)
   * [Prepare customized training data](#prepare_train_data)
   * [Start  demo for prediction](#pred_demo)
   * [Prepare customized prediction input](#prepare_pred_data)
   * [Generate ready to use spectral library](#gen_lib)
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

## <span id="down_pretain">Download pre-trained model parameters</span>



## <span id="use_gpu">Use GPU (optional)</span>

* DeepPhospho can be runned with GPU or with only CPU
* To use GPU, please confirm the information of your device and install suitable drive ([CUDA](https://developer.nvidia.com/cuda-downloads) (10.0 or 10.1 is recommended if you use the conda enviroment provided in this repository) and [cuDNN](https://developer.nvidia.com/cudnn))

# <span id="start">Quick start (from library/result file to DeepPhospho generated library in one command)</span>

* To generate a ready to use spectral library from an input data will always need the following steps:
  1. Convert the input training data to model compatible format
  2. Fine-tune the pre-trained model parameters to fit the under-analyzed data better
  3. Convert the input prediction data to model compatible format
  4. Predict the ion intensity and retention time of expected data
  5. Generate a spectral library
* To make DeepPhospho easier to use, we provide a all-in-one script to automatically transform initial data, train and select the best model parameter, predict, and generate the final library
* The cost time depends on the size of data and whether enables rt model ensembl. If you run this script with CPU only and have a large dataset to train models, we recommand to not use `-en`

## <span id="command">Command template</span>

* Before run this script, please activate the conda enviroment and set directory to DeepPhospho

  ```shell
  cd /path/of/DeepPhospho  # for windows cmd and prompt, type d: or e: to switch to expected drive first
  conda activate deep_phospho
  ```

* Below is a command template, and each argument will be introduced after it

  ```shell
  python run_deep_phospho.py -w ./WorkFolder -t task_name -tf ./msms.txt -tt MQ1.6 -pf ./evidence.txt Lib.xls -pt MQ1.6 SNLib -d 0 -en -m
  ```

## <span id="argu_intro">Introduction to arguments</span>

* -w is "work directory", which can also be passed as --work_dir
  * All operations will be performed in this directory, so-called work directory
  * If not passed, this will be {start_time}-DeepPhospho-WorkFolder
* -t is "task name", which can also be passed as --task_name
  * This will be added to all generated files or folders as an identifier
  * If not passed, this will be Task_{start_time}
* -tf is "train file", which can also be passed as --train_file
  * This should point to the path of expected data for model fine-tuning
  * The supported file is Spectronaut library (exported as .xls file) and search result from MaxQuant (msms.txt stored in txt folder)
* -tt is "train file type (source)", which can also be passed as --train_file_type
  * This value defines the format of train file, and the correspondence is
    * SNLib - Spectronaut library
    * MQ1.5 - MaxQuant msms.txt (with MQ version <= 1.5, the phospho-modification is annotated as "(ph)")
    * MQ1.6 - MaxQuant msms.txt (with MQ version >= 1.6, the phospho-modification is annotated as "(Phospho (STY))")
* -pf is "prediction file", which can also be passed as --pred_file
  * This one is able to pass in multi files, either `-pf fileA fileB fileC` or `-pf fileA -pf fileB -pf fileC` is valid, and the mix of these two ways is also fine, like `-pf fileA -pf fileB fileC`
* -pt is "prediction file type", which can also be passed as --pred_file_type
  * If multi files are passed to -pf, then the same number should also be passed to this one, and this also support the mix ways as -pf
  * The valid files of this argument are as following
    * SNLib - Spectronaut library
    * SNResult - Spectronaut search result
    * MQ1.5 - MaxQuant msms.txt or evidence.txt
    * MQ1.6 - MaxQuant msms.txt or evidence.txt
    * PepSN13 - Spectronaut 13+ peptide format like \_[Acetyl (Protein N-term)]M[Oxidation (M)]LSLS[Phospho (STY)]PLK\_
    * PepMQ1.5 - MaxQuant 1.5- peptide format like \_(ac)GS(ph)QDM(ox)GS(ph)PLRET(ph)RK\_
    * PepMQ1.6 - MaxQuant 1.6+ peptide format like \_(Acetyl (Protein N-term))TM(Oxidation (M))DKS(Phospho (STY))ELVQK\_
    * PepComet - Comet peptide format like n#DFM\*SPKFS@LT@DVEY@PAWCQDDEVPITM\*QEIR
    * PepDP - DeepPhospho used peptide format like *1ED2MCLK
  * [notice] the first four types are files generated by other softwares, and the last five types (Pep + xxx) is a tab-separated two columns file, has "sequence" to store the modified peptides with any previous format and "charge" to store the precursor charge
* -d is "used device", which can also be passed as --device
  * For training and prediction, this argument can be cpu to use CPU only, or 0 to use GPU0, 1 to use GPU1, ...
* -en is "use ensembl RT model", which can also be passed as --rt_ensembl
  * If passed, ensembl RT model will be used to improve the predicted RT accuracy
  * This will increase the RT model training time by 5 times accordingly
* -m is "merge all library to one", which can also be passed as --merge
  * If passed, a final library consist of all predicted data will be generated (the individual ones will also be kept)

# <span id="configs">DeepPhospho configs</span>

* In this section, we will introduce the configs for ion and rt models

## <span id="ion_config">Configs for ion intensity model</span>

* Here we use config_ion_model.py as an example
* WorkFolder can be set to 'Here' indicates the dir to run script, or other specific path
* ExpName is the experiment name of this time, it will be an identifier and empty is also fine
* InstanceName will fully overwrite the instance name which was defined as the combination of ExpName, DataName and some other information as default
* TaskPurpose can be set to one of 'Train' or 'Predict' (case is ignored)
* PretrainParam is used
  * as pre-trained parameter for fine-tuning, and it can be empty in training mode to train the parameters ab initio
  * as model parameter to load for predicting, and it must be pointed to an vailid path of parameter file
* Intensity_DATA_CFG
  * DataName is used as the identifier of this dataset
  * The two setting groups below will have one to be ignored according to the TaskPurpose
    1. for training
       * TrainPATH, TestPATH, and HoldoutPATH are used to train model, and Holdout can be empty
    2. for prediction
       * PredInputPATH is defined as the prediction input
       * InputWithLabel can be True or False. If True, the evaluation will be done if the label is provided in the prediction input
  * MAX_SEQ_LEN will limit the max peptide length for either training and prediction. Though it is possible to predict any peptide longer than this setting, we recommended to train a new model for the specific length
  * We use cache (pickle) to make the data loading more quickly, and refresh_cache will re-pickle the input data
* MODEL_CFG
  * For ion intensity model, only MODEL_CFG (LSTMTransformer) is available and please make sure the UsedModelCFG is set to this one
  * In json format, just change the values in UsedModelCFG
* TRAINING_HYPER_PARAM
  * GPU_INDEX can be set to '0', '1', '2', ... or 'cpu', and corresponded GPU device or CPU will be used
  * EPOCH can be set to positive integer, 30 is recommended for fine-tuning
  * BATCH_SIZE is recommended to be set as $2^n$ according to the memory of your device

## <span id="rt_config">Configs for RT model</span>

* Config for RT model is similar as ion model. Some ones different are listed below
* PretrainParam is only used for fine-tuning if it is provided. Instead, ParamsForPred will be used as the params for prediction to be loaded
* MIN_RT and MAX_RT are used to scale the input to 0-1 and unscale the output to this range
* To train or fine-tune RT model, MODEL_CFG (LSTMTransformer) will be used, and Ensemble_MODEL_CFG (LSTMTransformerEnsemble) is used for prediction
  * To train a RT model and in .py config mode, change num_encd_layer for MODEL_CFG and set UsedModelCFG to MODEL_CFG
  * To predict RT in .py config mode, change UsedModelCFG to Ensemble_MODEL_CFG and num_encd_layer will be ignored
  * To train a RT model in .json config mode, change num_encd_layer in UsedModelCFG and set model_name to 'LSTMTransformer'
  * To predict RT in .json config mode, set model_name to 'LSTMTransformerEnsemble'
* The ensembl is implemented by changing the num_encd_layer (number of transformer encoder layer) of each model, and we provided 4, 5, 6, 7, 8 five pre-trained parameters

## <span id="import_config">Use config file and overwrite values with command line arguments</span>

* We provide flexible ways to import configs

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

# <span id="manu">Train and predict manurally</span>

## <span id="train_demo">Start demo for model training</span>

* We provided a demo for ion intensity model and RT model fine-tuning based on our pre-trained parameters. And the dataset RPE1 DIA used in this demo is also the data for EGF phospho-signaling analysis in our paper
* Before start this, please make sure these files are existed and in correct format
  * In folder demo/RPE1_DIA_demo_data, two files for ion model and two files for RT model are existed and please unzip the zipped ones
  * For demo of ion intensity model, best_model.pth should exist in folder PretrainParams/IonModel
  * For demo of RT model, 4.pth should exist in folder PretrainParams/RTModel
* Below is the training steps
  * Open command line (prompt in windows or any shell in linux) and change the conda enviroment to deep_phospho
  * Change directory to DeepPhospho main folder
  * run `python train_ion.py ./demo/ConfigDemo-IonModel-RPE1_DIA-Finetune_ion_model.json` to start ion intensity model fine-tuning
    * [Notice] the GPU_IDX in this config file is set to "0", if you want to use cpu only or other device, please change it in the config file or run the following command instead `python train_ion.py -c ./demo/ConfigDemo-IonModel-RPE1_DIA-Finetune_ion_model.json -g cpu`
  * run `python train_rt.py ./demo/ConfigDemo-RTModel-RPE1_DIA-Finetune_rt_model.json` to start RT model fine-tuning based on pre-trained parameters with 4 encoder layer
    * [Notice] the GPU_IDX in this config file is also set to "0"
    * [Notice] we use ensembl model to improve the final performance of RT model, and we provided 5 pre-trained parameters with 4, 5, 6, 7, and 8 encoder layers. The num_encd_layer in this demo config is set to "4" and PretrainParam is "4.pth". To train the same five models, please create five config files and run them substantially, or use one config file and add arguments like `python train_rt.py -c ./demo/ConfigDemo-RTModel-RPE1_DIA-Finetune_rt_model.json -l 5 - p /path/to/5.pth` to fine-tune the RT model param with 5 encoder layers

## <span id="prepare_train_data">Prepare customized training data</span>

* The most obvious obstacles to run released deep learning models are usually the setup of enviroment and the preparation of data with compatible format for each specific model

* Here, we provided some functions to convert formats more easily

* For training data, we now support two formats

  * Spectronaut library (exported as plain text format)
  * MaxQuant search result msms.txt

* After preparation of one training data, run the following script

  ```shell
  python generate_dataset.py train -f library.xls -t SNLib -o ./output_folder
  ```

* As shown above, four arguments should passed to the script

  1. the first is train (the other one is pred and will be introduced in next part)
  2. -f is the expected training data
  3. -t is type or source of the given training data
     * SNLib - library from Spectronaut
     * MQ1.5 - msms.txt file from MaxQuant with version <= 1.5
     * MQ1.6 - msms.txt file from MaxQuant with version >= 1.6 (these two versions have different modified peptide format)
  4. -o is the output folder, here is not a path to output file because four files will be generated, including train and val datasets for both ion and RT models

* If you have data from other sources, you can contact us to add the support of your data format

* You can also generate datasets following these rules

* The dataset for ion model is a json file which will be loaded as a dict to use and modify easily

  * Each key of the dict is a peptide precursor, which as the format like @HEDGHESMVP2TYR.4, * or @ at first position means Acetyl modified or not, and 1, 2, 3, 4 indicate M(ox), S(ph), T(ph), Y(ph) respectively
  * Each value is also a dict to store the fragment-intensity pairs
  * The fragments have format like b5+1-Noloss, b5+1-1,NH3, b5+1-1,H2O, b5+1-1,H3PO4 for 5th b ion with 1 charge state and has no loss, loss 1 NH3, loss 1 H2O, loss 1 H3PO4
  * And the intensity can be any values without normalization (like relative intensity with 100 as max value)

## <span id="pred_demo">Start  demo for prediction</span>

* After running the above two training demos, there will be two folders created in the demo folder, which contain multi files including the trained parameters
* [Notice] In the training demos, we defined the "InstanceName" in config files to make the name of output folders consistent. In general, we recommand to fill in "ExpName" and "DataName" to auto create work folder, which will have start time and some information in the generated name
* run `python pred_ion.py ./demo/ConfigDemo-IonModel-RPE1_DIA-Pred_with_finetuned_parameteres.json` to predict spectra of some peptide precursors with the model parameters fine-tuned just now
* run `python pred_rt.py ./demo/ConfigDemo-RTModel-RPE1_DIA-Pred_with_finetuned_parameteres.json` to predict iRT of some peptides with the model parameters fine-tuned just now
* If you want to use CPU or other GPU device but not "0", add -c before config path and add '-g cpu' or '-g 1', '-g 2', ...

## <span id="prepare_pred_data">Prepare customized prediction input</span>

* We also provided the convertion funcstion for prediction input

* The usage is like training data but change train to pred

  ```shell
  python generate_dataset.py pred -f library.xls -t SNLib -o ./output_folder
  ```

* Here we support the following file formats

  1. SNLib - library from Spectronaut
  2. SNResult - search results from Spectronaut
  3. MQ1.5 - evidence.txt or msms.txt file from MaxQuant with version <= 1.5
  4. MQ1.6 - evidence.txt or msms.txt file from MaxQuant with version >= 1.6
  5. PepSN13 - Spectronaut 13+ peptide format like \_[Acetyl (Protein N-term)]M[Oxidation (M)]LSLS[Phospho (STY)]PLK\_
  6. PepMQ1.5 - MaxQuant 1.5- peptide format like \_(ac)GS(ph)QDM(ox)GS(ph)PLRET(ph)RK\_
  7. PepMQ1.6 - MaxQuant 1.6+ peptide format like \_(Acetyl (Protein N-term))TM(Oxidation (M))DKS(Phospho (STY))ELVQK\_
  8. PepComet - Comet peptide format like n#DFM\*SPKFS@LT@DVEY@PAWCQDDEVPITM\*QEIR
  9. PepDP - DeepPhospho used peptide format like *1ED2MCLK

* 1 - 4 is the file from Spectronaut or MaxQuant

* 5 - 9 is tab-separated file with two columns "sequence" and "charge", and total five peptide formats can be assigned for this file. This will be convenient if there is only a peptide list collected from any other data

* If you want to generate prediction input yourself, please following these rules

* For ion model, a single column file with title "IntPrec" and rows with precursor like *1ED2MCLK.2

* For RT model, a single column file with title "IntPep" and rows with peptide lieke *1ED2MCLK

## <span id="gen_lib">Generate ready to use spectral library</span>

* We provided a script to build library from DeepPhospho predicted results with ion intensity and RT
* run `python build_spec_lib.py build -i ion_result.json -r rt_result.txt -o output_library.xls`
* To merge multi (at least two) libraries to one
* run `python build_spec_lib.py merge -l libA libB libC -o output_library.xls`
* [notice] different with dataset generation with generate_dataset.py, the -o output here should be a file path since only one file will be generated

# <span id="license">License</span>

* DeepPhospho is under a general MIT license

# <span id="pub">Publication</span>






