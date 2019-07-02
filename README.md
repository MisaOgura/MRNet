# MRNet

[![DOI](https://zenodo.org/badge/192375650.svg)](https://zenodo.org/badge/latestdoi/192375650)

PyTorch implementation of the paper [Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699), published by the [Stanford ML Group](https://stanfordmlgroup.github.io/).

It is developed for participating in the [MRNet Competition](https://stanfordmlgroup.github.io/competitions/mrnet/). For more info, see [Background](#background) section.

## Citation

```txt
Misa Ogura. (2019, July 1). MisaOgura/MRNet: MRNet baseline model (Version 0.0.1). Zenodo. http://doi.org/10.5281/zenodo.3264923
```

## TL;DR - Quickstart

### 0. Clone the repo and `cd` into it

```terminal
$ git clone git@github.com:MisaOgura/MRNet.git
Cloning into 'MRNet'...
...
Resolving deltas: 100% (243/243), done.

$ cd MRNet
```

### 1. Setup an environment

The code is developed with `Python 3.6.8`.

The packages directly required are:

```txt
docopt==0.6.2
joblib==0.13.2
numpy==1.16.4
pandas==0.24.2
Pillow==6.0.0
scikit-learn==0.21.2
torch==1.1.0
torchvision==0.3.0
tqdm==4.32.1
```

Please make sure you have these packages with same minor versions available in your environment.

### 2. Download data

- Request access to the dataset on the [MRNet Competition](https://stanfordmlgroup.github.io/competitions/mrnet/) page

- Unzip the archive and save it to the `MRNet` project root

    ```terminal
    $ unzip -qq MRNet-v1.0.zip -d path/to/MRNet (./ if you are already in it)

    # Note that you will see some warnings - it seems ok to ignore it
    ```

- You now should have `MRNet-v1.0` data directory in the project root

    ```terminal
    $ cd path/to/MRNet

    $ tree -L 1
    .
    ├── LICENSE.txt
    ├── MRNet-v1.0
    ├── README.md
    ├── environment.yml
    ├── notebooks
    ├── scripts
    └── src
    ```

### 3. Merge diagnoses to make labels

Diagnoses (`0` for negative, `1` for positive) of each condition per case are provided as three separate `csv` files. It would be handy to have all the diagnoses per case in one place, so we will merge the three dataframes and save it as one `csv` file.

```terminal
$ python scripts/make_labels.py -h
Merges csv files for each diagnosis provided in the original dataset into
one csv per train/valid dataset.

Usage:
  make_labels.py <data_dir>
  make_labels.py (-h | --help)

General options:
  -h --help          Show this screen.

Arguments:
  <data_dir>         Path to a directory where the data lives e.g. 'MRNet-v1.0'
```

```terminal
$ python -u scripts/make_labels.py MRNet-v1.0
Parsing arguments...
Created 'train_labels.csv' and 'valid_labels.csv' in MRNet-v1.0
```

### 4. Train models

Now we're ready to move on to training!

#### 4.1. Train convolutional neural networks (CNNs)

First step is to train [9 CNNs](https://journals.plos.org/plosmedicine/article/figure?id=10.1371/journal.pmed.1002699.g002), each predicting probabilities of 3 diagnoses (abnormal, acl tear and meniscual tear) based on an MRI series from 3 planes (axial, sagittal and coronal).

```terminal
$ python src/train_cnn_models.py -h
Trains three CNN models to predict abnormalities, ACL tears and meniscal
tears for a given plane (axial, coronal or sagittal) of knee MRI images.

Usage:
  train_cnn_models.py <data_dir> <plane> <epochs> [options]
  train_cnn_models.py (-h | --help)

General options:
  -h --help             Show this screen.

Arguments:
  <data_dir>            Path to a directory where the data lives e.g. 'MRNet-v1.0'
  <plane>               MRI plane of choice ('axial', 'coronal', 'sagittal')
  <epochs>              Number of epochs e.g. 50

Training options:
  --lr=<lr>             Learning rate for nn.optim.Adam optimizer [default: 0.00001]
  --weight-decay=<wd>   Weight decay for nn.optim.Adam optimizer [default: 0.01]
  --device=<device>     Device to run code ('cpu' or 'cuda') - if not provided,
                        it will be set to the value returned by torch.cuda.is_available()
```

To train CNNs, run:

```terminal
$ python -u src/train_cnn_models.py MRNet-v1.0 axial 10
Parsing arguments...
Creating data loaders...
Creating models...
Training a model using axial series...
Checkpoints and losses will be save to ./models/2019-06-25_12-37
=== Epoch 1/10 ===
Train losses - abnormal: 0.257, acl: 1.168, meniscus: 0.906
Valid losses - abnormal: 0.272, acl: 0.747, meniscus: 0.769
Valid AUCs - abnormal: 0.853, acl: 0.765, meniscus: 0.657
Min valid loss for abnormal, saving the checkpoint...
Min valid loss for acl, saving the checkpoint...
Min valid loss for meniscus, saving the checkpoint...
=== Epoch 2/50 ===
...
```

It create a directory for each experiment, named with a timestamp `{datetime.now():%Y-%m-%d_%H-%M}`, e.g. `2019-06-25_12-37` where all the output will be stored.

A checkpoint `cnn_{plane}_{diagnosis}_{epoch:02d}.pt` is saved whenever the loweset validation loss is achieved for a particular diagnosis. The training and validation losses are also saved as `losses_{plane}.csv`.

#### 4.2. Train logistic regression models

For a given diagnosis, predictions from 3 series per exam are combined using [logistic regression](https://journals.plos.org/plosmedicine/article/figure?id=10.1371/journal.pmed.1002699.g004) to weight them accordingly and generate a single output for each exam in the training set.

```terminal
$ python src/train_lr_models.py -h
Trains logistic regression models for abnormalities, ACL tears and meniscal
tears, by combine predictions from CNN models.

Usage:
  train_lr_models.py <data_dir> <models_dir>
  train_lr_models.py (-h | --help)

General options:
  -h --help         Show this screen.

Arguments:
  <data_dir>        Path to a directory where the data lives e.g. 'MRNet-v1.0'
  <models_dir>      Directory where CNN models are saved e.g. 'models/2019-06-24_04-18'
```

To train logistic regression models, run:

```terminal
$ python -u src/train_lr_models.py MRNet-v1.0 path/to/models
Parsing arguments...
Loading CNN best models from path/to/models...
Creating data loaders...
Collecting predictions on train dataset from the models...
Training logistic regression models for each condition...
Cross validation score for abnormal: 0.661
Cross validation score for acl: 0.649
Cross validation score for meniscus: 0.689
Logistic regression models saved to path/to/models
```

Note that the code will look for the **best CNN checkpoints** saved in the `models_dir` by sorting each model and taking the *last one*. This is because in `src/train_cnn_models.py`, checkpoints are saved in a format `cnn_{plane}_{diagnosis}_{epoch:02d}.pt` every time the minimum validation loss is achieved. Hence the one with the **largest epoch value** per model is considered the best.

You will now have `lr_{diagnosis}.pkl` models saved to `path/to/models` directory, along with the checkpoints and losses.

### 5. Evaluate a model

We have trained 9 CNNs and 3 logistic regrssion models. Let's evaluate them.

#### 5.1. Obtain predictions

First we need to obtain model predictions on the validation dataset.

```terminal
$ python src/predict.py -h
Calculates predictions on the validation dataset, using CNN models specified
in src/cnn_models_paths.txt and logistic regression models specified in
src/lr_models_paths.txt

Usage:
  predict.py <valid_paths_csv> <output_dir>
  predict.py (-h | --help)

General options:
  -h --help          Show this screen.

Arguments:
  <valid_paths_csv>  csv file listing paths to validation set, which needs to
                     be in a specific order - an example is provided as
                     valid-paths.csv in the root of the project
                     e.g. 'valid-paths.csv'
  <output_dir>       Directory where predictions are saved as a 3-column csv
                     file (with no header), where each column contains a
                     prediction for abnormality, ACL tear, and meniscal tear,
                     in that order
                     e.g. 'out_dir'
```

We need to create `src/cnn_models_paths.txt` and `src/lr_models_paths.txt` to point
the programme to the right models. This is so that it is easier to test different combinations of models, when you have many models developed in separate experiments.

Models need to be listed in a specific order:

```terminal
$ cat src/cnn_models_paths.txt
path/to/models/cnn_sagittal_abnormal_{epoch:02d}.pt
path/to/models/cnn_coronal_abnormal_{epoch:02d}.pt
path/to/models/cnn_axial_abnormal_{epoch:02d}.pt
path/to/models/cnn_sagittal_acl_{epoch:02d}.pt
path/to/models/cnn_coronal_acl_{epoch:02d}.pt
path/to/models/cnn_axial_acl_{epoch:02d}.pt
path/to/models/cnn_sagittal_meniscus_{epoch:02d}.pt
path/to/models/cnn_coronal_meniscus_{epoch:02d}.pt
path/to/models/cnn_axial_meniscus_{epoch:02d}.pt
```

```terminal
$ cat src/lr_models_paths.txt
path/to/models/lr_abnormal.pkl
path/to/models/lr_acl.pkl
path/to/models/lr_meniscus.pkl
```

Once we create these 2 files, we're ready to proceed. To generate predictions on the `valid` dataset, run:

```terminal
$ python -u src/predict.py valid-paths.csv output/dir
Loading CNN models listed in src/cnn_models_paths.txt...
Loading logistic regression models listed in src/lr_models_paths.txt...
Generating predictions per case...
Predictions will be saved as output/dir/predictions.csv
```

The output should look like this (mock data):

```
7.547038087153214170e-02,1.751259132483399053e-02,2.848331082853714641e-02
2.114864409946341783e-01,2.631492356970821164e-02,3.936068787607087394e-02
3.527864673292197550e-01,2.275642573873807861e-01,4.486585856423670055e-02
4.285206463344938543e-02,1.557965692434650634e-02,2.385414339529156116e-02
4.834032069244934005e-01,4.263092724193431882e-02,3.172960607334367467e-01
```

#### 5.2. Calculate AUC scores

Finally, let's calculate the average AUC of the abnormality detection, ACL tear, and Meniscal tear tasks, which is the metrics reported on the [leaderboard](https://stanfordmlgroup.github.io/competitions/mrnet/).

```terminal
$ python src/evaluate.py -h
Calculates the average AUC score of the abnormality detection, ACL tear and
Meniscal tear tasks.

Usage:
  evaluate.py <valid_paths_csv> <preds_csv> <valid_labels_csv>
  evaluate.py (-h | --help)

General options:
  -h --help          Show this screen.

Arguments:
  <valid_paths_csv>    csv file listing paths to validation set, which needs to
                       be in a specific order - an example is provided as
                       valid-paths.csv in the root of the project
                       e.g. 'valid-paths.csv'
  <preds_csv>          csv file generated by src/predict.py
                       e.g. 'out_dir/predictions.csv'
  <valid_labels_csv>   csv file containing labels for the valid dataset
                       e.g. 'MRNet-v1.0/valid_labels.csv'
```

To calculate AUC scores, run:

```terminal
$ python -u src/evaluate.py valid-paths.csv path/to/predictions.csv MRNet-v1.0/valid_labels.csv
Reporting AUC scores...
  abnormal: 0.930
  acl: 0.865
  meniscus: 0.749
  average: 0.848
```

And there you have it!

### 6. Submitting the model for official evaluation

Once you have your model, you can submit it for an official evaluation by following the [tutorial](https://worksheets.codalab.org/worksheets/0xcaf785cb84564239b240400fbea93ec5/) provided by the authors.

**N.B.** Make sure to use `src/predict_codalab.py` which conforms to the API specification of the submittion process.

According to them it takes around 2 weeks for the score to appear on the leaderboard.

## Background

In the paper [Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699), the [Stanford ML Group](https://stanfordmlgroup.github.io/) developed an algorithm to predict abnormalities in knee MRI exams, and measured the clinical utility of providing the algorithm’s predictions to radiologists and surgeons during interpretation.

They developed a deep learning model for detecting:

- **general abnormalities**

- **anterior cruciate ligament (ACL)**

- **meniscal tears**

### MRNet Dataset description

The **dataset (~5.7G)** was released along with the publication of the paper. You can download it by agreeing to the Research Use Agreement and submitting your details on the [MRNet Competition](https://stanfordmlgroup.github.io/competitions/mrnet/) page.

It consists of **1,370 knee MRI exams**, containing:

- **1,104 (80.6%) abnormal exams**

- **319 (23.3%) ACL tears**

- **508 (37.1%) meniscal tears**

The dataset is split into:

- **training set (1,130 exams, 1,088 patients)**

- **validation set (120 exams, 111 patients)** - called _tuning set_ in the paper

- **hidden test set (120 exams, 113 patients)** - called _validation set_ in the paper

The hidden test set is _not publically available_ and is used for scoring models submitted for the competition.

**N.B.**

- [Stratified random sampling](https://en.wikipedia.org/wiki/Stratified_sampling) was used to ensure _at least 50 positive examples_ of abnormal, ACL tear and meniscal tear were preset in each set.

- All exams from each parient were put in the same split.

- In the paper, an external validation was performed on a [pubclically available data](http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/).

# Author

### Misa Ogura

#### 👩🏻‍💻 R&D Software Engineer @ [BBC](https://www.bbc.co.uk/rd/blog)

#### 🏳️‍🌈 Co-founder of [Women Driven Development](https://womendrivendev.org/)

[Github](https://github.com/MisaOgura) | [Medium](https://medium.com/@misaogura) | [twitter](https://twitter.com/misa_ogura) | [LinkedIn](https://www.linkedin.com/in/misaogura/)
