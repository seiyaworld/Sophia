+++
title = 'BERT'
date = 2024-06-16T10:38:59+09:00
draft = false
+++


## 構造
### Scaled Dot-Product Attention
長さnのトークンX。Wは線形変換を行う行列。\
クエリ$Q=XW_Q$,キー$K=XW_K$,バリュー$V=XW_V$\

行列積$QK^T$を$\sqrt{d_k}$で割る。次元が大きいと最終的にほとんどの重みがほぼ0になるため、$\sqrt{d_k}$で割る。\
そしてsoftmax関数を適用。$softmax(\frac{QK^T}{\sqrt{d_k}})$\


Attention = $softmax(\frac{QK^T}{\sqrt{d_k}})V$

### Multi-Head Attention
Scaled Dot-Product Attentionを集約。

### Residual Connection
入力(X)+出力(Multi-Head Attention)を次の処理に送る。

### Layer Nomalization
$\frac{\gamma}{\sigma}\bigodot(y-\mu)+\beta$
平均$\mu$と標準偏差$\sigma$

### Feedforward Network
GELU関数を適用する。

## 入力
### トークン
先頭に\[CLS\]、末尾に\[SEP\]を加える。
## 学習
事前学習
- マスク付き言語モデル
- Next Sentence Prediction
ファインチューニング

## ライブラリ
インストール
```
# colob
!pip install transformers==4.41.2 fugashi==1.3.2 ipadic==1.0.0
```

読み込み
```Python
import torch
from transformers import BertJapaneseTokenizer, BertModel
```

### トークナイザ
```Python
name = 'tohoku-nlp/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(name)
```

```Python
tokenizer.tokenize('私はこの世界で魔法を使用する。')
```

```Python
encodes = tokenizer.encode('私はこの世界で魔法を使用する。')
encodes
```

```Python
tokenizer.convert_ids_tokens(encodes)
```

