"""
Physics Transformer for Higgs Discovery Classification
-------------------------------------------------------
Multiclass classifier — number of classes is dynamic, passed via config.

Architecture:
- Physics-aware token grouping (muon, dimuon, jet, dijet, softjet tokens)
- Transformer encoder blocks with multi-head attention
- MLP classification head with softmax output
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Reshape, Concatenate,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model


# ── Token group indices (column positions in the 35-feature input) ──────────
# These match data_columns ordering exactly
MU1_IDX    = [0, 1]                          # mu1_eta, mu1_pt
MU2_IDX    = [2, 3]                          # mu2_eta, mu2_pt
DIMU_IDX   = [4, 5, 6, 7, 8, 9, 10, 11,     # dR_mumu ... pt_centrality
              12, 15, 16]
JET1_IDX   = [17, 18, 19]                    # j1_eta, j1_pt, j1_btagPNetQvG
JET2_IDX   = [20, 21, 22]                    # j2_eta, j2_pt, j2_btagPNetQvG
DIJET_IDX  = [23, 24, 25]                    # m_jj, delta_eta_jj, pt_jj
SOFTJET_IDX= [26, 27, 28, 29, 30, 31,        # nJet ... SoftActivityJetNjets10
              32, 33, 34]

TOKEN_GROUPS = [MU1_IDX, MU2_IDX, DIMU_IDX,
                JET1_IDX, JET2_IDX, DIJET_IDX, SOFTJET_IDX]
TOKEN_NAMES  = ['mu1', 'mu2', 'dimu', 'jet1', 'jet2', 'dijet', 'softjet']

NUM_FEATURES = 35  # length of data_columns


class GatherLayer(tf.keras.layers.Layer):
    """Selects a subset of feature columns to form one physics token."""
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = indices

    def call(self, x):
        return tf.gather(x, self.indices, axis=1)

    def get_config(self):
        config = super().get_config()
        config['indices'] = self.indices
        return config


def build_model(config: dict) -> tf.keras.Model:
    """
    Build physics transformer from a Ray Tune config dict.

    Expected config keys:
        embedding_dim       : int   — token embedding size
        num_heads           : int   — attention heads (must divide embedding_dim)
        num_blocks          : int   — transformer encoder blocks
        ff_multiplier       : int   — feedforward expansion factor
        dropout_embed       : float — dropout after token concat
        dropout_attn        : float — dropout inside attention
        dropout_head        : float — dropout in MLP head
        lr                  : float — Adam learning rate
        weight_decay        : float — AdamW weight decay
    """
    embedding_dim  = config['embedding_dim']
    num_heads      = config['num_heads']
    num_blocks     = config['num_blocks']
    ff_mult        = config['ff_multiplier']
    drop_embed     = config['dropout_embed']
    drop_attn      = config['dropout_attn']
    drop_head      = config['dropout_head']

    inp = Input(shape=(NUM_FEATURES,), name='physics_input')

    # ── Token embedding ───────────────────────────────────────────
    token_list = []
    for i, idxs in enumerate(TOKEN_GROUPS):
        tok = GatherLayer(idxs, name=f'gather_{TOKEN_NAMES[i]}')(inp)
        tok = Dense(embedding_dim, name=f'embed_{TOKEN_NAMES[i]}')(tok)
        tok = Reshape((1, embedding_dim))(tok)
        token_list.append(tok)

    tokens = Concatenate(axis=1)(token_list)         # (batch, 7, embedding_dim)
    tokens = Dropout(drop_embed)(tokens)

    # ── Transformer encoder blocks ────────────────────────────────
    for b in range(num_blocks):
        attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=drop_attn,
            name=f'attn_{b}'
        )(tokens, tokens)
        tokens = LayerNormalization(name=f'ln1_{b}')(tokens + attn)

        ff = Dense(embedding_dim * ff_mult, activation='gelu', name=f'ff1_{b}')(tokens)
        ff = Dropout(drop_attn)(ff)
        ff = Dense(embedding_dim, name=f'ff2_{b}')(ff)
        tokens = LayerNormalization(name=f'ln2_{b}')(tokens + ff)

    # ── Classification head ───────────────────────────────────────
    x = GlobalAveragePooling1D()(tokens)

    x = Dense(128, activation='gelu')(x)
    x = LayerNormalization()(x)
    x = Dropout(drop_head)(x)

    x = Dense(64, activation='gelu')(x)
    x = LayerNormalization()(x)
    x = Dropout(drop_head / 2)(x)

    out = Dense(config['num_classes'], activation='softmax', name='output')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=config['lr'],
            weight_decay=config['weight_decay']
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model