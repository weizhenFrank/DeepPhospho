import copy

RT_DATA_PREPROCESS_CATALOG = dict(
    U2OS_RT=dict(
        MIN_RT=-100,
        MAX_RT=200,

        # MIN_RT=-60,
        # MAX_RT=160,
        # change MIN and MAX for testing influence of MIN and MAX
        MAX_SEQ_LEN=52,
    ),

    DDA_RT={
        # "MIN_RT": -60,
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
