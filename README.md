# deepRank

Deisgn sgRNA for CRISPR/Cas9 gene editing by deep learning.

Requirements:

1. Python3.5
2. Tensorflow 0.12.1
3. numpy
4. scipy

Sample Code:

```
from deep_rank import *
import numpy as np
import pandas as pd

# model path
model_save_path = './crispr_ko_conv2_hidden1.ckpt'

# load sample data
deep_rank_input = pd.read_csv('./sample_data.csv')

# preprocess data
deep_rank_input.loc[:, 'rank_score'] = 0
deep_rank_input = generate_input_from_clean_df(deep_rank_input)
deep_rank_input_x, _ = transform(deep_rank_input, seq_len=34)

# compute DeepRank Score
deep_rank_score, loss = predict(model_save_path, deep_rank_input_x, _, inference_fun=model_conv2_hidden1)
``` 