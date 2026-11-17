# Heart Disease Classification with a Decision Tree from Scratch

A Python project that implements a Decision Tree classifier from scratch
using **NumPy**, then applies it to the Cleveland Heart Disease dataset from
the **UCI Machine Learning Repository**.

> **Academic project:** developed as part of the Introduction to Artificial
> Intelligence course at the Federal University of Fronteira Sul (UFFS).
> The project is educational and is not a medical diagnostic tool.

## Project overview

The goal was to understand an existing Decision Tree implementation, adapt
it to a different dataset, and evaluate its classification performance.

Instead of using a ready-made Decision Tree estimator from a machine-learning
library, the tree-building logic in `src/decision_tree.py` is implemented
using NumPy. Scikit-learn is used only for splitting the dataset and
calculating evaluation metrics.

## What the project does

1. Downloads the Cleveland Heart Disease dataset from UCI.
2. Replaces missing-value markers (`?`) with missing values and removes
   incomplete records.
3. Converts the original target into a binary classification:
   - `0`: no heart disease indicated by the dataset.
   - `1`: heart disease indicated by the dataset.
4. Splits the data into training (80%) and test (20%) sets.
5. Trains a Decision Tree using entropy and information gain.
6. Evaluates predictions on the test set and prints the learned tree.

## Algorithm

The model selects feature thresholds by evaluating the 25th, 50th and 75th
percentiles of each feature. For each candidate split, it calculates the
weighted entropy of the resulting groups. The split with the greatest
information gain is selected.

The tree stops growing when it reaches the configured maximum depth, when
a node contains only one class, when no valid split exists, or when the
information gain does not exceed the configured minimum.

## Technologies

- Python
- NumPy
- Pandas
- Scikit-learn (train/test split and evaluation metrics)

## Repository structure

```text
heart-disease-decision-tree/
├── data/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── decision_tree.py
│   └── evaluation.py
├── tests/
│   └── test_decision_tree.py
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/heart-disease-decision-tree.git
cd heart-disease-decision-tree
```

### 2. Create a virtual environment (recommended)

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the experiment

```bash
python main.py
```

An internet connection is required the first time the dataset is loaded.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

## Experiment configuration

| Parameter | Value |
|---|---:|
| Training split | 80% |
| Test split | 20% |
| Random state | 42 |
| Maximum tree depth | 4 |
| Minimum samples per leaf | 2 |
| Split criterion | Information gain (entropy) |

## Results

The original academic report recorded **81.67% accuracy** on a test set of
60 records, using an 80/20 split and `random_state=42`.

The refactored implementation in this repository should be run to obtain its
own result. Small implementation differences can change the learned tree and
the final accuracy; do not treat the number in the academic report as a
guaranteed result of every version.

## Limitations and responsible use

- This is an educational classification experiment, not a clinically
  validated model.
- The dataset is relatively small and may not represent all populations.
- Accuracy alone does not establish clinical usefulness.
- The model must not be used to diagnose, screen, or make treatment
  decisions about real patients.

## Data source

Dua, D. and Graff, C. (2019). *UCI Machine Learning Repository*.
Heart Disease Dataset. https://doi.org/10.24432/C52P4X

Original implementation studied:

Enozeren. *Building a Decision Tree from Scratch*.
https://medium.com/@enozeren/building-a-decision-tree-from-scratch-324b9a5ed836

Related source repository:
https://github.com/enozeren/DecisionTree

## Academic context

Work developed for the course **Introduction to Artificial Intelligence**,
Bachelor's Degree in Artificial Intelligence, Federal University of
Fronteira Sul (UFFS), Realeza campus.

The repository is a reorganized and documented portfolio version of the
academic work. The Decision Tree implementation was based on the work by
Enozeren linked above; the author and source should remain credited. No
license is declared for this repository because the reuse terms of the
original implementation should be checked before assigning one. Review the
code and README before publishing, and make sure the final repository
accurately reflects your own contribution and understanding.
