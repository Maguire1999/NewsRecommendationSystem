# News Recommendation

## Presentation
![](demo/Web.png)
<video src="demo/demo.mp4" width="600px" height="600px" controls="controls"></video>


The repository currently includes the following models.

**Models in published papers**

| Model     | Full name                                                                 | Paper                                              |
| --------- | ------------------------------------------------------------------------- | -------------------------------------------------- |
| NRMS      | Neural News Recommendation with Multi-Head Self-Attention                 | https://www.aclweb.org/anthology/D19-1671/         |


## Get started to train

Basic setup.

```bash
git clone https://github.com/yusanshi/NewsRecommendation
cd NewsRecommendation
pip3 install -r requirements.txt
```

Download and preprocess the data.

```bash
mkdir data && cd data
# Download GloVe pre-trained word embedding
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
sudo apt install unzip
unzip glove.840B.300d.zip -d glove
rm glove.840B.300d.zip

# Download MIND dataset
# By downloading the dataset, you agree to the [Microsoft Research License Terms](https://go.microsoft.com/fwlink/?LinkID=206977). For more detail about the dataset, see https://msnews.github.io/.

# Uncomment the following lines to use the MIND Large dataset (Note MIND Large test set doesn't have labels, see #11)
# wget https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip
# unzip MINDlarge_train.zip -d train
# unzip MINDlarge_dev.zip -d val
# unzip MINDlarge_test.zip -d test
# rm MINDlarge_*.zip

# Uncomment the following lines to use the MIND Small dataset (Note MIND Small doesn't have a test set, so we just copy the validation set as test set :)
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip MINDsmall_train.zip -d train
unzip MINDsmall_dev.zip -d val
cp -r val test # MIND Small has no test set :)
rm MINDsmall_*.zip

# Preprocess data into appropriate format
cd ..
python3 src/data_preprocess.py
# Remember you shoud modify `num_*` in `src/config.py` by the output of `src/data_preprocess.py`
```

Modify `src/config.py` to select target model. The configuration file is organized into general part (which is applied to all models) and model-specific part (that some models not have).

```bash
vim src/config.py
```

Run.

```bash
# Train and save checkpoint into `checkpoint/{model_name}/` directory
python3 src/train.py
# Load latest checkpoint and evaluate on the test set
python3 src/evaluate.py
```

You can visualize metrics with TensorBoard.

```bash
tensorboard --logdir=runs

# or
tensorboard --logdir=runs/{model_name}
# for a specific model
```

> Tip: by adding `REMARK` environment variable, you can make the runs name in TensorBoard more meaningful. For example, `REMARK=num-filters-300-window-size-5 python3 src/train.py`.


### Optim study in MIND-mini

| Model     | AUC | MRR | nDCG@5 | nDCG@10 | Remark |
| --------- | --- | --- | ------ | ------- | ------ |
| baseline  | 0.6253    |   0.2823   |     0.3051   |   0.3731      |        |
| +SGD      |   0.5188   |    0.2148  |    0.2250    |     0.2905     |        |
| +AdamW      |   0.6298   |    0.2841  |    0.3091    |     0.3765     |        |


### Norm study in MIND-mini

| Model     | AUC | MRR | nDCG@5 | nDCG@10 | Remark |
| --------- | --- | --- | ------ | ------- | ------ |
| baseline  | 0.6253    |   0.2823   |     0.3051   |   0.3731      |        |
| +BN      |   0.5252   |    0.2476  |    0.2565    |     0.3181     |        |
| +GN     |    0.6323 |  0.2884   |   0.3122     |    0.3795     |        |
| +IN       | 0.6321    |  0.2847   |    0.3101     |    0.3785     |        |
| +LN       | 0.6404    |  0.2905   |    0.3172     |    0.3835     |        |


### Results in MIND-mini
| Model     | AUC | MRR | nDCG@5 | nDCG@10 | Remark |
| --------- | --- | --- | ------ | ------- | ------ |
| baseline  | 0.6253    |   0.2823   |     0.3051   |   0.3731      |        |
| +LN  +AdamW  + Cosine decay   | 0.6421    |  0.2960   |    0.3239     |    0.3890     |        |



## Get started to open website
```bash
cd ..
python3 src/web.py
```