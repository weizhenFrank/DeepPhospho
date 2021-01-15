
Intensity_DATA_PREPROCESS_CFG = {
    'MAX_SEQ_LEN': 74,
    'repeat_factor': 4,
    'mask_ratio': 1,
}

Intensity_DATA_CFG = {
    'DataName': '',

    "Intensity_FIELD_NAME": "normalized_intensity",
    "SEQUENCE_FIELD_NAME": 'sequence',
    "PRECURSOR_CHARGE": 'charge',

    "DATA_PROCESS_CFG": Intensity_DATA_PREPROCESS_CFG,
    'refresh_cache': True,

    "TrainPATH": "./data/to_pred/20201219-IntenInput-For_PhosDIA_DIA18.txt",
    "TestPATH": "./data/to_pred/20201219-IntenInput-For_PhosDIA_DIA18.txt",
    "HoldoutPATH": "./data/to_pred/20201219-IntenInput-For_PhosDIA_DIA18.txt",
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

UsedModelCFG = MODEL_CFG

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
    # interval(iteration) of saving the the parameters
    save_param_interval=300,
    GPU_INDEX='1',
    module_namelist=None,
    remove_ac_pep=False,  # here to remove peptide of ac in N terminal
    # MSE L1 PearsonLoss SALoss SA_Pearson_Loss L1_SA_Pearson_Loss SALoss_MSE
    # RMSE
    loss_func="MSE",
    LR_STEPS=(2000, 6000),
    BATCH_SIZE=128,
    EPOCH=10,

    use_prosit_pretrain=False,
    # this means we first to predict whether it exists for each fragment under
    # physical rule, then do regression.
    two_stage=True,
    lambda_cls=0.0,  # set two stage, this hyper parameter determine the weights of cls loss
    pdeep2mode=True,
    inter_layer_prediction=False,
    add_phos_principle=True,
    use_all_data=False,
    only_two_ions=False,
    pretrain_param='.checkpoint/ion_inten/ion_inten-R2P2-LSTMTransformer-RemoveSigmoidRemove0AssignEpoch90OfJeffVeroE6-remove_ac_pepFalse-add_phos_principleTrue-LossTypeMSE-use_holdoutFalse-2020_10_28_23_38_04/ckpts/best_model.pth',
)

TEST_HYPER_PARAM = {
    "Use multiple iteration": False
}
