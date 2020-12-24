# DeepPhospho
the code of DeepPhospho

## 1. installation 
### 1.1 DeepPhospho installation
    git clone https://github.com/weizhenFrank/DeepPhospho.git
### 1.2 conda environment installation
    conda env create -f RT_Ion.yml
## Train model on RT task
* Go to the file PutOnGitHub/IonInten/configs/config_main.py 
* change the mode = 'RT'
* chaneg the GPU_INDEX according to your machine in TRAINING_HYPER_PARAM
* change the data_name = "U2OS_DIA" (one of  "U2OS_DIA_RT" "PhosDIA_DIA18_finetune_RT" "PhosDIA_DDA_RT")
* (optional) use the checkpoint in checkpoints folder, and change pretrain_param='<checkpoint path>' in RT_TRAINING_HYPER_PARAM
* run the main.py
        python main.py
    

