# Attribution and provenance

The STGCN backbone, graph operators, data-loading pipeline, and early-stopping code in this project are derived from the [hazdzz/STGCN](https://github.com/hazdzz/STGCN) PyTorch implementation. Its GNU Lesser General Public License, version 2.1, is reproduced unchanged in [LICENSE](LICENSE).

The underlying STGCN architecture is described in:

Bing Yu, Haoteng Yin, and Zhanxing Zhu. *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting*. IJCAI, 2018. [Paper](https://arxiv.org/abs/1709.04875).

The temporal representation builds on:

Seyed Mehran Kazemi et al. *Time2Vec: Learning a Vector Representation of Time*. 2019. [Paper](https://arxiv.org/abs/1907.05321).

## Modifications in this research project

The HND research extends the reference implementation with daily and weekday Time2Vec streams, harmonic initialization, and positive learnable frequencies parameterized as `omega_base * exp(log_scale)`. The streams are projected and concatenated with the speed input. The final training recipe uses dedicated optimizer parameter groups for frequencies and phases.

This publication contains a cleaned implementation of that final configuration, training instructions, historical result CSVs, an offline prediction dashboard, and utilities for locally trained weights. It is a modified research implementation, not an unmodified upstream release. Earlier NMF/adaptive-graph, discrete-embedding, Huber-loss, and EMA experiments are outside this publication.

Pretrained model checkpoints are not distributed. Historical result tables, dashboard predictions, and temporal-parameter CSVs are included as research and presentation artifacts.

## Dataset provenance

PeMSD7-M originates from the Caltrans Performance Measurement System benchmark used by the original STGCN project. The bundled `vel.csv` and `adj.npz` were carried forward from the reference implementation's PeMSD7-M dataset. The original dataset distribution is linked in the [README](README.md#data-and-reproducibility).

The upstream software license is included for the derived code; it is not a declaration of independent licensing rights over third-party datasets or referenced publications.
