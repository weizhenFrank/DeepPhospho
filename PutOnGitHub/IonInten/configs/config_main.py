from IonInten.configs.data_config import DATA_CATALOG as _DATA_CATALOG
from IonInten.configs.model_config import MODEL_CFG_CATALOG as _MODEL_CFG_CATALOG


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


TRAIN_DATA_CFG = _DATA_CATALOG['%s_train' % data_name]
TRAIN_DATA_CFG['refresh_cache'] = refresh_cache
if '%s_test' % data_name not in _DATA_CATALOG:
    TEST_DATA_CFG = TRAIN_DATA_CFG
else:
    TEST_DATA_CFG = _DATA_CATALOG['%s_test' % data_name]

if '%s_holdout' % data_name not in _DATA_CATALOG:
    HOLDOUT_DATA_CFG = TRAIN_DATA_CFG
else:
    HOLDOUT_DATA_CFG = _DATA_CATALOG['%s_holdout' % data_name]

TEST_DATA_CFG['refresh_cache'] = refresh_cache
HOLDOUT_DATA_CFG['refresh_cache'] = refresh_cache

DATA_PROCESS_CFG = _DATA_CATALOG['%s_train' % data_name]['DATA_PROCESS_CFG']

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
    pretrain_param='.checkpoint/IonInten/IonInten-R2P2-LSTMTransformer-RemoveSigmoidRemove0AssignEpoch90OfJeffVeroE6-remove_ac_pepFalse-add_phos_principleTrue-LossTypeMSE-use_holdoutFalse-2020_10_28_23_38_04/ckpts/best_model.pth',)


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
MODEL_CFG.update(_MODEL_CFG_CATALOG[MODEL_CFG['model_name']])
