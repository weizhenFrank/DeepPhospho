import copy


RT_DATA_PREPROCESS_CATALOG = {
    "MIN_RT": -100,
    "MAX_RT": 200,
    "MAX_SEQ_LEN": 74,
}

refresh_cache = True

RT_DATA_CATALOG = {
    'DataName': '',
    'Train': {
        "data_path": "./data/to_pred/",
        "data_fn": "20201219-RTInput-For_PhosDIA_DIA18.txt",
        "RT_mode": True,
        "To_Predict": True,
        "SEQUENCE_FIELD_NAME": 'IntPep',
        "RT_FIELD_NAME": "iRT",
        "SCALE_BY_ZERO_ONE": True,
        "DATA_PROCESS_CFG": RT_DATA_PREPROCESS_CATALOG,
        'refresh_cache': refresh_cache,
    },
    "Test": {
        "data_path": "./data/RT/",
        "data_fn": "20201010-RT_Val-PhosDIA-DDA-seed0_811.txt",
        "RT_mode": True,
        "SEQUENCE_FIELD_NAME": 'IntPep',
        "RT_FIELD_NAME": "iRT",
        "SCALE_BY_ZERO_ONE": True,
        "DATA_PROCESS_CFG": RT_DATA_PREPROCESS_CATALOG,
        'refresh_cache': refresh_cache,

    },
    "Holdout": {
        "data_path": "./data/RT/",
        "data_fn": "20201010-RT_Test-PhosDIA-DDA-seed0_811.txt",
        "RT_mode": True,
        "SEQUENCE_FIELD_NAME": 'IntPep',
        "RT_FIELD_NAME": "iRT",
        "SCALE_BY_ZERO_ONE": True,
        "DATA_PROCESS_CFG": RT_DATA_PREPROCESS_CATALOG,
        'refresh_cache': refresh_cache,

    },
}
DATA_CATALOG = copy.deepcopy(RT_DATA_CATALOG)

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
    num_encd_layer=2,
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
    transformer_hidden_dim=1024,
)

TRAINING_HYPER_PARAM = dict(
    # MSE L1 PearsonLoss SALoss SA_Pearson_Loss L1_SA_Pearson_Loss SALoss_MSE
    # RMSE
    loss_func="RMSE",
    BATCH_SIZE=64,
    EPOCH=60,
    LR_STEPS=(10000, 20000),
    add_hydro=False,
    # this means we first to predict whether it exists for each fragment under
    # physical rule, then do regression.
    add_rc=False,
    pretrain_param='',

    resume=False,
    Bert_pretrain=False,  # mask language models training MLM
    accumulate_mask_only=False,  # loss agregation style for MLM
    DEBUG=False,  # less training data for debugging
    lr_scheduler_type='WarmupMultiStepLR',  # Noam, WarmupMultiStepLR
    LR=1e-4,
    transformer_on_epoch=-1,
    factor=0.2,  # the scheduler curve of Noam LR
    warmup_steps=5000,  # warm up_steps for Noam LR scheduler
    weight_decay=1e-8,
    warmup_iters=0,  # warm up_steps for other scheduler
    # interval(iteration) of saving the the parameters
    save_param_interval=300,
    GPU_INDEX='1',
    module_namelist=None,
    remove_ac_pep=False,  # here to remove peptide of ac in N terminal
)

TEST_HYPER_PARAM = {
    "Use multiple iteration": False
}
