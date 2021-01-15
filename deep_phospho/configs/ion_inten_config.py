import copy


RT_DATA_PREPROCESS_CATALOG = dict(
    U2OS_RT={
        "MIN_RT": -100,
        "MAX_RT": 200,
        "MAX_SEQ_LEN": 52,
    },

    DDA_RT={
        "MIN_RT": -100,
        "MAX_RT": 200,
        "MAX_SEQ_LEN": 74,
    },

    DIA18_RT={
        "MIN_RT": -100,
        "MAX_RT": 200,
        "MAX_SEQ_LEN": 52,
    },

    PredInput_RT={
        "MIN_RT": -100,
        "MAX_RT": 200,
        "MAX_SEQ_LEN": None,
    },

)


MODEL_CFG_CATALOG = dict(
    LSTMTransformer=dict(
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
        num_encd_layer=2,  # change to 1, 2, 3, 4, 5, 6, 7, 8, 9 for model ensemble (original 8)
        transformer_hidden_dim=1024,

    ),

    LSTMTransformerEnsemble=dict(
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
)


Intensity_PREPROCESS_CATALOG = {

    "PredInput": dict(
        MAX_SEQ_LEN=None,
    ),
    "U2OS": dict(
        MAX_SEQ_LEN=52,
    ),


    "DDA": dict(
        MAX_SEQ_LEN=74,
        # pretrain hyper param
        repeat_factor=4,
        mask_ratio=1,
    ),

    "DIA18": dict(
        MAX_SEQ_LEN=52,
        # pretrain hyper param
        repeat_factor=4,
        mask_ratio=1,
    ),
    "DIA185": dict(
        MAX_SEQ_LEN=52,
        # pretrain hyper param
        repeat_factor=4,
        mask_ratio=1,
    )}


DATA_PREPROCESS_CATALOG = copy.deepcopy(Intensity_PREPROCESS_CATALOG)
DATA_PREPROCESS_CATALOG.update(RT_DATA_PREPROCESS_CATALOG)

RT_DATA_CATALOG = {

    "PredInput_RT_train": {
        "data_path": "./data/to_pred/",
        "data_fn": "20201219-RTInput-For_PhosDIA_DIA18.txt",
        "RT_mode": True,
        "To_Predict": True,
        "SEQUENCE_FIELD_NAME": 'IntPep',
        "RT_FIELD_NAME": "iRT",
        "SCALE_BY_ZERO_ONE": True,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['PredInput_RT']
    },

    "PhosDIA_DDA_RT_train": {
        "data_path": "./data/RT/",
        "data_fn": "20201010-RT_Train-PhosDIA-DDA-seed0_811.txt",
        "RT_mode": True,
        "SEQUENCE_FIELD_NAME": 'IntPep',
        "RT_FIELD_NAME": "iRT",
        "SCALE_BY_ZERO_ONE": True,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['DDA_RT']
    },
    "PhosDIA_DDA_RT_test": {
        "data_path": "./data/RT/",
        "data_fn": "20201010-RT_Val-PhosDIA-DDA-seed0_811.txt",
        "RT_mode": True,
        "SEQUENCE_FIELD_NAME": 'IntPep',
        "RT_FIELD_NAME": "iRT",
        "SCALE_BY_ZERO_ONE": True,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['DDA_RT']
    },
    "PhosDIA_DDA_RT_holdout": {
        "data_path": "./data/RT/",
        "data_fn": "20201010-RT_Test-PhosDIA-DDA-seed0_811.txt",
        "RT_mode": True,
        "SEQUENCE_FIELD_NAME": 'IntPep',
        "RT_FIELD_NAME": "iRT",
        "SCALE_BY_ZERO_ONE": True,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['DDA_RT']
    },

    "PhosDIA_DIA18_finetune_RT_train": {
        "data_path": "./data/RT/",
        "data_fn": "20201010-RT_Train-PhosDIA-DIA18-seed0_811.txt",
        "RT_mode": True,
        "SEQUENCE_FIELD_NAME": 'IntPep',
        "RT_FIELD_NAME": "iRT",
        "SCALE_BY_ZERO_ONE": True,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['DIA18_RT']
    },
    "PhosDIA_DIA18_finetune_RT_test": {
        "data_path": "./data/RT/",
        "data_fn": "20201010-RT_Val-PhosDIA-DIA18-seed0_811.txt",
        "RT_mode": True,
        "SEQUENCE_FIELD_NAME": 'IntPep',
        "RT_FIELD_NAME": "iRT",
        "SCALE_BY_ZERO_ONE": True,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['DIA18_RT']
    },
    "PhosDIA_DIA18_finetune_RT_holdout": {
        "data_path": "./data/RT/",
        "data_fn": "20201010-RT_Test-PhosDIA-DIA18-seed0_811.txt",
        "RT_mode": True,
        "SEQUENCE_FIELD_NAME": 'IntPep',
        "RT_FIELD_NAME": "iRT",
        "SCALE_BY_ZERO_ONE": True,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['DIA18_RT']
    },

    "U2OS_DIA_RT_train": {
        "data_path": "./data/RT/",
        "data_fn": "20201010-RT_Train-U2OS-DIA-seed0_811.txt",
        "RT_mode": True,
        "SEQUENCE_FIELD_NAME": 'IntPep',
        "RT_FIELD_NAME": "iRT",
        "SCALE_BY_ZERO_ONE": True,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['U2OS_RT']
    },
    "U2OS_DIA_RT_test": {
        "data_path": "./data/RT/",
        "data_fn": "20201010-RT_Val-U2OS-DIA-seed0_811.txt",
        "RT_mode": True,
        "SEQUENCE_FIELD_NAME": 'IntPep',
        "RT_FIELD_NAME": "iRT",
        "SCALE_BY_ZERO_ONE": True,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['U2OS_RT']
    },
    "U2OS_DIA_RT_holdout": {
        "data_path": "./data/RT/",
        "data_fn": "20201010-RT_Test-U2OS-DIA-seed0_811.txt",
        "RT_mode": True,
        "SEQUENCE_FIELD_NAME": 'IntPep',
        "RT_FIELD_NAME": "iRT",
        "SCALE_BY_ZERO_ONE": True,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['U2OS_RT']
    },

}

Intensity_DATA_CATALOG = {
    "PredInput_train":
        {
            "data_path": "./data/to_pred/",
            "data_fn": "20201219-IntenInput-For_PhosDIA_DIA18.txt",
            "To_Predict": True,
            "SEQUENCE_FIELD_NAME": 'IntPrec',
            "Intensity_FIELD_NAME": "normalized_intensity",
            "PRECURSOR_CHARGE": 'charge',
            "REMOVE_OUT_RANGE": False,
            "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['PredInput'],
        },
    "U2OS_DIA_train": {
        "data_path": "./data/IonIntensity/",
        "data_fn": "20201010-Inten_Train-U2OS-DIA-seed0_811.json",
        "To_Predict": False,
        "Intensity_FIELD_NAME": "normalized_intensity",
        "SEQUENCE_FIELD_NAME": 'sequence',
        "PRECURSOR_CHARGE": 'charge',
        "FOR_TRAINING": True,
        "REMOVE_OUT_RANGE": False,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['U2OS']
    },
    "U2OS_DIA_test": {
        "data_path": "./data/IonIntensity/",
        "data_fn": "20201010-Inten_Val-U2OS-DIA-seed0_811.json",
        "To_Predict": False,
        "Intensity_FIELD_NAME": "normalized_intensity",
        "SEQUENCE_FIELD_NAME": 'sequence',
        "PRECURSOR_CHARGE": 'charge',
        "FOR_TRAINING": False,
        "REMOVE_OUT_RANGE": False,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['U2OS']},
    "U2OS_DIA_holdout": {
        "data_path": "./data/IonIntensity/",
        "data_fn": "20201010-Inten_Test-U2OS-DIA-seed0_811.json",
        "To_Predict": False,
        "Intensity_FIELD_NAME": "normalized_intensity",
        "SEQUENCE_FIELD_NAME": 'sequence',
        "PRECURSOR_CHARGE": 'charge',
        "FOR_TRAINING": False,
        "REMOVE_OUT_RANGE": False,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['U2OS']},

    "PhosDIA_DIA18_train": {
        "data_path": "./data/IonIntensity/",
        "data_fn": "20201010-Inten_Train-PhosDIA-DIA18-seed0_811.json",
        "To_Predict": False,
        "Intensity_FIELD_NAME": "normalized_intensity",
        "SEQUENCE_FIELD_NAME": 'sequence',
        "PRECURSOR_CHARGE": 'charge',
        "FOR_TRAINING": True,
        "REMOVE_OUT_RANGE": False,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['DIA18']
    },
    "PhosDIA_DIA18_test": {
        "data_path": "./data/IonIntensity/",
        "data_fn": "20201010-Inten_Val-PhosDIA-DIA18-seed0_811.json",
        "To_Predict": False,
        "Intensity_FIELD_NAME": "normalized_intensity",
        "SEQUENCE_FIELD_NAME": 'sequence',
        "PRECURSOR_CHARGE": 'charge',
        "FOR_TRAINING": False,
        "REMOVE_OUT_RANGE": False,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['DIA18']},
    "PhosDIA_DIA18_holdout": {
        "data_path": "./data/IonIntensity/",
        "data_fn": "20201010-Inten_Test-PhosDIA-DIA18-seed0_811.json",
        "To_Predict": False,
        "Intensity_FIELD_NAME": "normalized_intensity",
        "SEQUENCE_FIELD_NAME": 'sequence',
        "PRECURSOR_CHARGE": 'charge',
        "FOR_TRAINING": False,
        "REMOVE_OUT_RANGE": False,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['DIA18']},

    "PhosDIA_DDA_train": {
        "data_path": "./data/IonIntensity/",
        "data_fn": "20201010-Inten_Train-PhosDIA-DDA-seed0_811.json",
        "To_Predict": False,
        "Intensity_FIELD_NAME": "normalized_intensity",
        "SEQUENCE_FIELD_NAME": 'sequence',
        "PRECURSOR_CHARGE": 'charge',
        "FOR_TRAINING": True,
        "REMOVE_OUT_RANGE": False,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['DDA']
    },
    "PhosDIA_DDA_test": {
        "data_path": "./data/IonIntensity/",
        "data_fn": "20201010-Inten_Val-PhosDIA-DDA-seed0_811.json",
        "To_Predict": False,
        "Intensity_FIELD_NAME": "normalized_intensity",
        "SEQUENCE_FIELD_NAME": 'sequence',
        "PRECURSOR_CHARGE": 'charge',
        "FOR_TRAINING": False,
        "REMOVE_OUT_RANGE": False,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['DDA']},
    "PhosDIA_DDA_holdout": {
        "data_path": "./data/IonIntensity/",
        "data_fn": "20201010-Inten_Test-PhosDIA-DDA-seed0_811.json",
        "To_Predict": False,
        "Intensity_FIELD_NAME": "normalized_intensity",
        "SEQUENCE_FIELD_NAME": 'sequence',
        "PRECURSOR_CHARGE": 'charge',
        "FOR_TRAINING": False,
        "REMOVE_OUT_RANGE": False,
        "DATA_PROCESS_CFG": DATA_PREPROCESS_CATALOG['DDA']},

}

DATA_CATALOG = copy.deepcopy(Intensity_DATA_CATALOG)
DATA_CATALOG.update(RT_DATA_CATALOG)


Mode = "RT"  # IonIntensity RT

data_name = "U2OS_DIA_RT"

# U2OS_DIA PhosDIA_DIA18  PhosDIA_DDA
# PredInput


# for RT
# U2OS_DIA_RT PhosDIA_DIA18_finetune_RT PhosDIA_DDA_RT
# PredInput_RT

refresh_cache = True
Use_cache = True
Save_cache = True


TRAIN_DATA_CFG = DATA_CATALOG['%s_train' % data_name]
TRAIN_DATA_CFG['refresh_cache'] = refresh_cache
if '%s_test' % data_name not in DATA_CATALOG:
    TEST_DATA_CFG = TRAIN_DATA_CFG
else:
    TEST_DATA_CFG = DATA_CATALOG['%s_test' % data_name]

if '%s_holdout' % data_name not in DATA_CATALOG:
    HOLDOUT_DATA_CFG = TRAIN_DATA_CFG
else:
    HOLDOUT_DATA_CFG = DATA_CATALOG['%s_holdout' % data_name]

TEST_DATA_CFG['refresh_cache'] = refresh_cache
HOLDOUT_DATA_CFG['refresh_cache'] = refresh_cache

DATA_PROCESS_CFG = DATA_CATALOG['%s_train' % data_name]['DATA_PROCESS_CFG']

TRAINING_HYPER_PARAM = dict(
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
    save_param_interval=300,  # interval(iteration) of saving the the parameters
    GPU_INDEX='1',
    module_namelist=None,
    remove_ac_pep=False,  # here to remove peptide of ac in N terminal
)

TEST_HYPER_PARAM = {
    "Use multiple iteration": False
}

Intensity_TRAINING_HYPER_PARAM = dict(
    loss_func="MSE",  # MSE L1 PearsonLoss SALoss SA_Pearson_Loss L1_SA_Pearson_Loss SALoss_MSE RMSE
    LR_STEPS=(2000, 6000),
    BATCH_SIZE=128,
    EPOCH=10,
    LR=1e-4,
    use_prosit_pretrain=False,
    two_stage=True,  # this means we first to predict whether it exists for each fragment under physical rule, then do regression.
    lambda_cls=0.0,  # set two stage, this hyper parameter determine the weights of cls loss
    pdeep2mode=True,
    inter_layer_prediction=False,
    add_phos_principle=True,
    use_all_data=False,
    only_two_ions=False,
    pretrain_param='.checkpoint/ion_inten/ion_inten-R2P2-LSTMTransformer-RemoveSigmoidRemove0AssignEpoch90OfJeffVeroE6-remove_ac_pepFalse-add_phos_principleTrue-LossTypeMSE-use_holdoutFalse-2020_10_28_23_38_04/ckpts/best_model.pth',)


RT_TRAINING_HYPER_PARAM = dict(
    loss_func="RMSE",  # MSE L1 PearsonLoss SALoss SA_Pearson_Loss L1_SA_Pearson_Loss SALoss_MSE RMSE
    BATCH_SIZE=64,
    EPOCH=60,
    LR_STEPS=(10000, 20000),
    add_hydro=False,
    add_rc=False,  # this means we first to predict whether it exists for each fragment under physical rule, then do regression.
    pretrain_param=''
)

if Mode == "RT":
    TRAINING_HYPER_PARAM.update(RT_TRAINING_HYPER_PARAM)

elif Mode == "IonIntensity":
    TRAINING_HYPER_PARAM.update(Intensity_TRAINING_HYPER_PARAM)
else:
    raise RuntimeError("No valid mode name given.")

MODEL_CFG = {
    "model_name": "LSTMTransformer",
    # LSTMTransformer  LSTMTransformerEnsemble
}
MODEL_CFG.update(MODEL_CFG_CATALOG[MODEL_CFG['model_name']])
