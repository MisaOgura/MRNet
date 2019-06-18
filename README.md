# MRNet Competition

## Background

In the paper [Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699), the [Stanford ML Group](https://stanfordmlgroup.github.io/) developed an algorithm to predict abnormalities in knee MRI exams, and measured the clinical utility of providing the algorithm’s predictions to radiologists and surgeons during interpretation.

They developed a deep learning model for detecting:

- **general abnormalities**

- **anterior cruciate ligament (ACL)**

- **meniscal tears**

## MRNet Dataset description

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

## Data preprocessing

The information on preprocessing can be found in the [Methods section](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699#sec008) of the paper.

The MRNet dataset consits of following components.

```bash
data/MRNet-v1.0
├── train
│   ├── axial  # plane
│   │   ├── 0000.npy  # case
│   │   ├── 0001.npy  # case
│   │   ...
│   ├── coronal
│   └── sagittal
├── train-abnormal.csv  # labels for abnormalities per case
├── train-acl.csv  # labels for ACL tear per case
├── train-meniscus.csv  # labels for meniscal tear per case
├── valid
│   ├── axial
│   ├── coronal
│   └── sagittal
├── valid-abnormal.csv
├── valid-acl.csv
└── valid-meniscus.csv
```

There are couple of things we need to do to prepare the dataset in the shape and format we want.

### 1. Organise image data per case

In the original data structure, the image data for `train` and `valid` sets is organised _per plane_. However, later on we will be combining the decisions obtained from each plane per case to produce a final decision. Therefore it would be more useful to organise the data per case.

### 2. Convert image data from `numpy` arrays to RGB image

Each series of MRI images is provided as a `<case_number>.npy` file, which contains a numpy array of shape `(s, 256, 256)` where s is the number of images in the series. We need to convert the image data to be an RGB image with 3 colour channel as an input to a CNN. We will save the converted image as a `png` file.

### 3. Merge diagnoses to make labels

The diagnosis (`0` for negative, `1` for positive) of each condition for the examined cases are provided as three separate `csv` files. It would be handy to have all the disgnoses per case in one place, so we will merge the three dataframes and save it as one `csv` file.

### Data processing scripts

You can process data as described above in one command.

From the root of the project, run:

```bash
$ ./scripts/process-data.sh <data_dir> <out_dir>

...
...
...
Preprocessing finished.
```

- `data_dir`: points to the root of the MRNet dataset e.g. `data/MRNet-v1.0`

- `out_dir`: where the process data should be stored e.g. `data/processed`

The output directory (~8.4G) looks like this:

```bash
data/processed
├── train
│   ├── 0000  # case
│   │   ├── axial  # plane
│   │   │   ├── 000.png  # image in a series
│   │   │   ├── 001.png
│   │   │   ...
│   │   ├── coronal
│   │   └── sagittal
│   ├── 0001
│   │   ├── axial
│   │   ├── coronal
│   │   └── sagittal
│   ...
├── train_labels.csv  # labels for all conditions per case
├── valid
└── valid_labels.csv
```
