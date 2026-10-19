# ML for Intrusion Detection — NSL-KDD

CYBER 363 project (Spring 2026): five models, four classical and one neural network, trained to
tell normal network connections from attacks in the NSL-KDD benchmark, and scored on how many
attacks they catch and how many normal connections they flag.

Revised after the course. The code and report as submitted are the tag `first-version`; what
changed, and why, is at the end of this file.

## Results

Each model is trained on 90% of `KDDTrain+` and scored on `KDDTest+`. The other 10% of the
training file is held out to choose a second threshold: the lowest one that flags no more than 1%
of its normal connections. Both are reported. From `results/metrics/summary.csv`:

| Model | Threshold | Attacks caught | seen types | new types | False alarms | Accuracy |
|---|---|---|---|---|---|---|
| Random Forest | default (0.5) | 60.5% | 76.8% | 21.1% | 2.7% | 76.4% |
| | 1% budget (0.077) | **90.0%** | 91.0% | **87.5%** | 9.9% | **90.0%** |
| SVM (RBF) | default (margin 0) | 68.0% | 77.2% | 45.6% | 7.2% | 78.7% |
| | 1% budget | 68.5% | 77.4% | 47.1% | 7.3% | 78.9% |
| Logistic Regression | default (0.5) | 62.6% | 71.3% | 41.5% | 7.4% | 75.5% |
| | 1% budget (0.75) | 60.5% | 69.8% | 38.0% | 7.0% | 74.5% |
| k-Nearest Neighbors | default (0.5) | 65.0% | 73.9% | 43.2% | 7.2% | 77.0% |
| | 1% budget (0.2) | 71.9% | 79.3% | 54.0% | 7.6% | 80.7% |
| Neural Network | default (0.5) | 69.3% | 76.4% | 52.2% | 6.7% | 79.7% |
| | 1% budget (0.39) | 70.3% | 77.1% | 53.7% | 7.0% | 80.0% |

*Attacks caught* is the share of attack connections flagged; *false alarms* the share of normal
connections flagged. *New types* are the 17 attack types (3,750 of the test file's 12,833 attacks)
that never appear in the training file, such as `mscan`, `apache2`, `processtable` and `snmpguess`.

What the table says:

- **The test file is the hard part, not the models.** Scored by 5-fold cross-validation inside the
  training file, the random forest is 99.9% accurate and logistic regression 97.4%. On the test file
  every model falls to 75 to 80% at its default threshold, because 29% of its attacks are of types
  none of them were trained on, and some types that were (`guess_passwd`: 53 training rows, 1,231
  test rows) are badly under-represented.
- **The random forest ranks new attacks above normal traffic, but rarely above 0.5.** Its ROC AUC
  is 0.96, the highest. Most of the trees call a new attack normal, so at the default threshold it
  catches 21% of them; at the threshold chosen from the training side's own false alarms (an
  attack score of 0.077) it catches 87.5%.
- **That costs false alarms.** The budget is 1% on the held-out training rows, but the test file's
  normal traffic differs too, and 9.9% of it is flagged, against 2.7% at 0.5. Which point is
  right depends on how many alerts an analyst can read. The other models barely move, because
  their own 1% point sits near their default.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

The NSL-KDD files are in `data/nslkdd/` (from the University of New Brunswick's
[Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html); cite
Tavallaee et al., 2009, if you use them).

## Running

```bash
OMP_NUM_THREADS=4 python main.py   # full run, about 5 minutes (the SVM is most of it)
python main.py --quick             # 10% of the data, under a minute
python main.py --skip-svm          # everything but the SVM
python main.py --reprocess         # redo preprocessing even if processed arrays exist
pytest                             # 8 tests, a few seconds
```

Results go to `results/metrics/summary.csv` and `results/plots/` (`detection.png`,
`roc_curves.png`, `training_time_bar.png`). `results.txt` is the log of the run above.

## Project structure

```
data/nslkdd/            KDDTrain+.txt, KDDTest+.txt
data/processed/full/    processed arrays (generated; quick runs use data/processed/quick/)
src/
  data_preprocessing.py one-hot encoding, scaling fitted on training rows, the validation slice,
                        and which test attacks are of new types
  metrics.py            attacks caught, false alarms, the false-alarm-budget threshold
  classical_models.py   random forest, SVM, logistic regression, k-nearest neighbours
  deep_learning_model.py the PyTorch network
  evaluation.py         summary table and plots
tests/                  known-answer tests
report/report.md        the report, revised
main.py
```

## What changed from the first version

- **"Recall 0.80" was not the share of attacks caught.** Precision, recall and F1 were averaged
  over the two classes, so the random forest's "recall 0.8008" was the mean of 97% of normal
  connections passed and 63% of attacks caught. The report read it as detection. Now attacks
  caught and false alarms are reported per class, split by seen and new attack types.
- **The SVM's 1,056 s of "training" was mostly calibration and testing.** `probability=True` ran a
  hidden 5-fold calibration (the fit alone took 808 s with it, 108 s without, measured separately),
  and the timer also covered predicting the test file. The calibration changed no prediction and
  no AUC. It is gone, and fitting and testing are timed apart.
- **`--quick` corrupted the next full run.** It saved its 10% arrays where the full run looked for
  processed data, so the next `python main.py` silently trained on 10%; and when full arrays
  existed, `--quick` was ignored. Quick and full arrays now live in separate folders.
- **Thresholds.** The report suggested tuning the random forest's threshold. It is now done without
  touching the test file: on a 10% slice held out of the training file, for a 1% false-alarm
  budget. The neural network uses the same slice for early stopping.
- **The report** said logistic regression's lower score "confirms the decision boundary is not
  linearly separable"; inside the training file it is 97.4% accurate, so the gap is the test file's
  new attacks. The report is revised with the measured numbers.
- The README asked for data that is already in the repository and listed a notebook and a models
  folder that did not exist; `results.txt` carried local paths.

The models were all trained on the whole training file in the first version and on 90% of it
now, so their default-threshold numbers differ slightly from the first report's.
