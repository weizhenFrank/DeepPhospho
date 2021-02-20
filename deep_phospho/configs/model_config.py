# all kings of models structure hyper-parameters

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
