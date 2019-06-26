# MRNet

Code implementation of the paper [Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699), the [Stanford ML Group](https://stanfordmlgroup.github.io/).

It is developed for participating in the [MRNet Competition](https://stanfordmlgroup.github.io/competitions/mrnet/).

For more info, see [Background](#background) section.

## TL;DR - Quickstart

### 1. Clone the repo

```terminal
$ git clone git@github.com:MisaOgura/MRNet.git
Cloning into 'MRNet'...
...
Resolving deltas: 100% (243/243), done.

$ cd MRNet

$ ls
README.md  notebooks/  scripts/  src/
```

### 2. Download data

- Request access to the dataset on the [MRNet Competition](https://stanfordmlgroup.github.io/competitions/mrnet/) page

- Unzip the archive and save it to the `MRNet` project root

    ```terminal
    $ unzip -qq MRNet-v1.0.zip -d path/to/MRNet
    ```

- You now should have `MRNet-v1.0` directory in the project root

    ```terminal
    $ cd path/to/MRNet

    $ ls
    README.md  notebooks/  scripts/  src/  MRNet-v1.0/
    ```

### 3. Merge diagnoses to make labels

Diagnoses (`0` for negative, `1` for positive) of each condition per case are provided as three separate `csv` files. It would be handy to have all the diagnoses per case in one place, so we will merge the three dataframes and save it as one `csv` file.

```terminal
$ python3 scripts/make_labels.py MRNet-v1.0
...
Created 'train_labels.csv' and 'valid_labels.csv' in MRNet-v1.0
```

Now we're ready to move onto training!

### 4. Train models

#### 4.1. Train convolutional neural networks (CNNs)

First step is to train [9 CNNs](https://journals.plos.org/plosmedicine/article/figure?id=10.1371/journal.pmed.1002699.g002), each predicting probabilities of 3 diagnoses (abnormal, acl tear and meniscual tear) based on an MRI series from 3 planes (axial, sagittal and coronal).

`src/train_mrnet.py` expects below parameters:

- `<data_dir>`: directory where the data lives
- `<plane>`: `axial`, `coronal` or `sagittal`
- `<epochs>`: number of epochs to train for
- `<lr>`: learning rate for `nn.optim.Adam` optimizer
- `<weight_decay>`: weight decay for `nn.optim.Adam` optimizer
- `<device>`: `cpu` or `cuda`

To train CNNs, run below from the project root:

```terminal
$ export PYTHONPATH=$PYTHONPATH:`pwd`

$ python3 -u src/train_mrnet.py MRNet-v1.0 axial 10 0.00001 0.01
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

It create a directory for each experiment, named with a timestamp (format: `f'{datetime.now():%Y-%m-%d_%H-%M}'`), where all the output will be saved.

A checkpoint is saved whenever the loweset validation loss is achieved for a particular diagnosis. The training and validation losses are also saved as a `csv` file.

#### 4.2. Train logistic regression models

For a given diagnosis, predictions from 3 series per exam are combined using [logistic regression](https://journals.plos.org/plosmedicine/article/figure?id=10.1371/journal.pmed.1002699.g004) to weight them accordingly and generate a single output for each exam in the training set.

`src/train_lr.py` expects below parameters:

- `<data_dir>`: directory where the data lives
- `<models_dir>`: directory where CNN models are saved

To train logistic regression models,

```terminal
$ python3 -u src/train_lr.py MRNet-v1.0 path/to/models
```

### 5. Evaluate a model

#### 5.1. Obtain predictions

#### 5.2. Calculate AUC scores

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
