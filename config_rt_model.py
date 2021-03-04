
# If the work folder defined as Here, the folder will be where the config file in
# Else define a specific folder
WorkFolder = 'Here'

# Experiment name will be used as the part of result folder name and plays as an identifier
ExpName = ''

# Instance name has higher priority than ExpName, and this will fully overwrite the name of result folder
InstanceName = ''

# TaskPurpose = 'Train'
TaskPurpose = 'Predict'

PretrainParam = './PretrainParams/RTModel/4.pth'

# If use ensembl - A dict with key-value pairs as Encoder layer number: param path
# If use single model - Str of param path
ParamsForPred = {
    4: "./PretrainParams/RTModel/4.pth",
    5: "./PretrainParams/RTModel/5.pth",
    6: "./PretrainParams/RTModel/6.pth",
    7: "./PretrainParams/RTModel/7.pth",
    8: "./PretrainParams/RTModel/8.pth",
}

RT_DATA_CFG = {
    'DataName': '',

    'TrainPATH': "./Data/RT_TestData/20201010-RT_Train-RPE1_DIA-seed0_811.txt",
    'TestPATH': "./Data/RT_TestData/20201010-RT_Test-RPE1_DIA-seed0_811.txt",
    'HoldoutPATH': "./Data/RT_TestData/20201010-RT_Holdout-RPE1_DIA-seed0_811.txt",

    "PredInputPATH": "./demo/RTInput.txt",
    'InputWithLabel': False,

    "SEQUENCE_FIELD_NAME": 'IntPep',
    "RT_FIELD_NAME": "iRT",
    "SCALE_BY_ZERO_ONE": True,

    "DATA_PROCESS_CFG": {
        "MIN_RT": -100,
        "MAX_RT": 200,
        "MAX_SEQ_LEN": 52,
    },
    'refresh_cache': False,
    'Use_cache': True,
}

MODEL_CFG = dict(
    model_name='LSTMTransformer',
    embed_dim=256,
    lstm_hidden_dim=512,
    lstm_layers=2,
    lstm_num=2,  # change to 1, 3 for model ensemble (original 2)
    bidirectional=True,
    max_len=100,
    num_attention_head=8,
    fix_lstm=False,
    pos_encode_dropout=0.1,
    attention_dropout=0.1,
    num_encd_layer=4,
    # change to 1, 2, 3, 4, 5, 6, 7, 8, 9 for model ensemble (original 8)
    transformer_hidden_dim=1024,
)

Ensemble_MODEL_CFG = dict(
    model_name='LSTMTransformerEnsemble',
    embed_dim=256,
    lstm_hidden_dim=512,
    lstm_layers=2,
    lstm_num=2,  # change to 1, 3 for model ensemble (original 2)
    bidirectional=True,
    max_len=100,
    num_attention_head=8,
    fix_lstm=False,
    pos_encode_dropout=0.1,
    attention_dropout=0.1,
    num_encd_layer=None,
    transformer_hidden_dim=1024,
)

UsedModelCFG = Ensemble_MODEL_CFG

TRAINING_HYPER_PARAM = dict(
    GPU_INDEX='0',
    EPOCH=2,
    BATCH_SIZE=64,

    # MSE L1 PearsonLoss SALoss SA_Pearson_Loss L1_SA_Pearson_Loss SALoss_MSE
    # RMSE
    loss_func="RMSE",
    LR_STEPS=(10000, 20000),
    add_hydro=False,
    # this means we first to predict whether it exists for each fragment under
    # physical rule, then do regression.
    add_rc=False,

    resume=False,
    Bert_pretrain=False,  # mask language models training MLM
    accumulate_mask_only=False,  # loss agregation style for MLM
    DEBUG=False,  # less training data for debugging
    lr_scheduler_type='WarmupMultiStepLR',  # Noam, WarmupMultiStepLR
    LR=1e-4,
    transformer_on_epoch=-1,
    factor=0.2,  # the scheduler curve of Noam LR
    warmup_factor=1 / 3,  # the scheduler curve of Noam LR
    warmup_steps=5000,  # warm up_steps for Noam LR scheduler
    weight_decay=1e-8,
    warmup_iters=0,  # warm up_steps for other scheduler
    # interval(iteration) of saving the the parameters
    save_param_interval=300,
    module_namelist=None,
    remove_ac_pep=False,  # here to remove peptide of ac in N terminal
)

TEST_HYPER_PARAM = {
    "Use multiple iteration": False
}
