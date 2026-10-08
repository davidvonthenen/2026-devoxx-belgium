# SVTRv2 + CTC training for Gaussian-blurred text

This project trains a PyTorch recognizer directly from the dataset created by
`create_gaussian_text_dataset.py`:

```text
after/*.png -> SVTRv2 encoder -> feature-rearrangement head -> character logits
                                                              |
manifest.csv: text ----------------------------------------> CTC loss
```

`after_path` is the model input. `text` is the character target. The clear images
in `before/` remain reference material and are never loaded by the training loop.
This model predicts text. It does not generate a reconstructed image.

## Files

```text
svtrv2_ctc_training/
├── model.py                  # SVTRv2 encoder and feature-rearrangement/CTC head
├── data.py                   # Manifest, images, vocabulary, and split checks
├── train.py                  # Default grid search, single runs, checkpoints, evaluation
├── predict.py                # Portable CPU/MPS/CUDA loading and prediction
├── tests/test_checkpoints.py  # Artifact/selection/loading tests and hardware-gated checks
├── create_gaussian_text_dataset.py
├── flatten.py
├── dataset/                  # The five supplied image pairs and manifest
├── README.md
```

The supplied ZIP contains checkpoint regression tests but does not include
`requirements.txt` or `NOTICE.md`. This update does not reconstruct those
dependency or attribution files. The install commands below assume your existing
project dependency file is available.

There is no pretrained model download, PaddlePaddle dependency, external training
framework, or custom CUDA extension. Training starts from random initialization.

## Architecture scope

The implementation follows the CTC-only OpenOCR SVTRv2 architecture:

- Two strided convolutions embed the image.
- Grouped-convolution blocks perform local mixing. Global self-attention blocks
  mix the two-dimensional visual features.
- The feature-rearrangement head applies row-wise attention and learned attention
  pooling over height, then classifies each horizontal position.

The local mixer uses two grouped 3x3 convolutions, matching the
`SVTRv2LNConvTwo33` design. Encoder residual blocks use post-normalization; the
row-wise attention block in the head uses pre-normalization. This is not an
arbitrary Transformer encoder with the SVTRv2 name attached.

**This is a fixed-size, grayscale, CTC-only adaptation.** It does not implement
the full paper's multi-size resizing (MSR) training strategy or its auxiliary
semantic guidance module (SGM). OpenOCR provides a CTC-only `svtrv2_rctc.yml`
configuration alongside its SGM training configuration. No published benchmark
accuracy is claimed for this adaptation. See [NOTICE.md](NOTICE.md).

Two configurations are available:

| Configuration | Stage dimensions | Stage depths | Parameters with ASCII vocabulary |
|---|---|---|---:|
| `compact` (default) | 64, 128, 192 | 3, 4, 3 | 3,206,464 |
| `reference` | 128, 256, 384 | 6, 6, 6 | 19,754,784 |

`reference` selects upstream stage widths/depths, not a pretrained checkpoint or
the complete paper training recipe. Both configurations use this project's
state-dictionary names and are not drop-in loaders for OpenOCR checkpoints.
Start with `compact` for the controlled blur experiment.

For a 512 x 64 image, the tensor path is:

```text
Input                 [B,   1, 64, 512]
Patch embedding       [B,  64, 16, 128]   compact dimensions
Encoder output        [B, 192,  8, 128]
Feature rearrangement [B, 128, 192]
CTC logits            [B, 128,  96]       95 ASCII characters + blank
```

All inputs retain their original geometry. There is no hidden resize to a
word-sized OCR input. Height must be divisible by 8 and width by 4. Both training
and prediction reject dimensions that differ from the saved model configuration.

## 1. Install

Use Python 3.11 or newer. Create an environment in this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

`requirements.txt` includes every direct training/inference dependency: PyTorch,
NumPy, Pillow, and safetensors. PyTorch remains pinned to the documented 2.10.0
baseline. No torchvision, torchaudio, or separate MPS package is required.
For CUDA or Linux CPU-only installations, select the matching PyTorch wheel
first as shown below, then install the requirements.

### NVIDIA H100/H200 training

The following selects the published PyTorch 2.10.0 CUDA 12.8 wheel. The host must
have a compatible NVIDIA driver. Other supported CUDA wheel choices are listed
on the official [PyTorch versions page](https://pytorch.org/get-started/previous-versions/).

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_bf16_supported())"
```

## 2. Dataset format

Extract your generated dataset without changing its internal layout:

```bash
unzip /path/to/output.zip -d data
```

This archive creates `data/output/`:

```text
data/output/
├── manifest.csv
├── skipped.csv
├── before/
│   └── 00000001.png
└── after/
    └── 00000001.png
```

Example manifest header and record:

```csv
sample_id,source_line,text,before_path,after_path,width,height,blur_radius
00000001,1,When did Beyonce start becoming popular?,before/00000001.png,after/00000001.png,512,64,3.0
```

The reader preserves spaces, case, and punctuation. It converts input images to
grayscale and normalizes each pixel using `pixel / 127.5 - 1.0`. It applies no
additional blur, augmentation, clipping, resizing, or text truncation. Missing
files, corrupt files, inconsistent sizes, unknown characters, duplicate sample
IDs, and invalid CTC alignments produce errors rather than being ignored.

## 3. Training, validation, and test splits

The automatic split targets **70% training, 15% validation, and 15% test rows**.
Duplicate text, including case and whitespace variants, stays in one group.
The normalization affects only the grouping key, not the training target.
Groups are seeded, ordered by size, and assigned to the partition with the
largest remaining row-count deficit. All three partitions must be non-empty.
Whole groups are never cut to enforce an exact percentage, so the achieved
fractions can differ when groups are large or the dataset is small.
Each grid trial writes `splits_split_XX.json` with sample IDs and the
fixed training-evaluation subset, where `XX` is the outer round number (`01`, `02`).
It also writes `splits_split_XX.txt`, a UTF-8, tab-separated text file listing
the actual image paths for each bucket. Explicit single-run mode keeps the
original `splits.json` and `splits.txt` names:

```text
bucket      sample_id  after_path          before_path
train       00000003   after/00000003.png  before/00000003.png
validation  00000001   after/00000001.png  before/00000001.png
test        00000002   after/00000002.png  before/00000002.png
```

The example shows selected rows, not a complete split. Each accepted manifest
sample appears exactly once in the full text file. Paths are relative to
`--data-dir`; `after_path` is the model input, and `before_path` is included for
reference when present in the manifest. No image files are moved or copied.
The text file spells the bucket `validation`; the internal/JSON name remains
`val`. The training-evaluation probe is a subset of `train`, not a fourth bucket.

Grid search repeats the full parameter grid on two partitions by default, with a cap of five.
One frozen partition is shared by all combinations within a round. Membership
can change between rounds, but the requested fractions do not. The search root
contains a combined `splits.txt` with additional `split_round` and `split_seed`
columns. Root `splits.json` is now an index of round directories, split hashes,
counts, and skipped duplicate rounds, rather than one global partition. Each
round entry includes `split_files` paths and JSON/text membership hashes. Each
`split_XX/` directory contains that round's `splits_split_XX.json` and
`splits_split_XX.txt`; its trial directories retain byte-identical copies,
including the fixed probe. Batch-order shuffling does not change partition membership.

`--validation-fraction` defaults to `0.15`. Changing this existing option adjusts
the validation share; the automatic test share remains `0.15`, and training
receives the remainder. At least three independent groups are required.

For existing corpus-level partitions, add a `split` column with `train`, `val`,
or `test` on every row. The reader preserves those assignments and their ratios
instead of creating a random split. Explicit partitions must include all three
sets. A manifest containing only train/val labels now fails with a clear error
rather than proceeding without a test score. During grid search, explicit
assignments run through the parameter grid once. They are never overridden to
manufacture additional split rounds.

## 4. Train on the larger generated dataset

**Grid search is the default.** The following command runs both outer split
rounds and every parameter combination without requiring a mode flag:

```bash
python train.py
```

Edit `param_grid` near the top of `train.py` to change the searched values.
The defaults are two rounds, 12 combinations per round, up to 50 epochs
per model, and patience five. `--grid-search` remains accepted for existing
commands; `--no-grid-search` is the explicit opt-out for one model.

The startup log prints `Execution mode: GRID SEARCH` and the scheduled
round/combination/trial counts. A deliberate single run prints `SINGLE RUN`
and states that the parameter grid and outer loop are disabled. Use a new
or empty output directory; existing results are never silently overwritten.

### Deliberately train one configuration

```bash
python train.py \
  --data-dir data/output \
  --run-dir runs/gaussian_svtrv2 \
  --no-grid-search \
  --model-size compact \
  --device cuda \
  --precision bf16 \
  --batch-size 32 \
  --eval-batch-size 8 \
  --num-workers 4 \
  --epochs 50 \
  --min-epochs 1 \
  --warmup-epochs 3 \
  --patience 5 \
  --min-delta 0.0001 \
  --learning-rate 0.0003 \
  --weight-decay 0.01
```

This command expects more than the five demonstration samples. Add the
smoke-test override only when deliberately testing the pipeline on small data.
Each experiment needs a new or empty `--run-dir`; previous runs are not
silently overwritten.

The batch sizes are starting settings, not measured H100/H200 capacity limits.
Reduce them for wider images or the larger model. Evaluation uses FP32 and has
its own smaller batch size. Training is single-process, single-device;
multi-GPU distributed training is not implemented.

The encoder uses PyTorch's scaled dot-product attention, which chooses a
supported implementation for the active device. There is no mandatory
`flash-attn` package. CUDA training uses BF16 autocast when requested; model
parameters, optimizer state, log-softmax, and CTC loss stay in FP32. BF16 does
not require the FP16 gradient-scaling path. This implementation intentionally
does not expose FP16 training.

### Search training hyperparameters

By default, `train.py` trains every combination in the `param_grid` dictionary
near the top of the file. `--grid-search` remains supported but is not required.
Use `--no-grid-search` only for a deliberate single run with one partition.
The adjacent `SPLIT_ROUNDS` constant is set to `2`; edit it
to an integer from `1` through `5` to change the number of rounds. Values outside that
range, booleans, and non-integers are rejected before training.

```python
param_grid: Dict[str, List[Any]] = {
    "learning_rate": [1e-4, 3.25e-4, 6.5e-4],
    "batch_size": [2, 4],
    "weight_decay": [0.02, 0.05],
}
```

The grid contains **12 combinations per split round**. With two distinct
partitions, this produces **24 sequential training runs**, not parallel models.
Each configuration creates one model and trains that same instance for up to
**50 epochs**, with early-stopping patience **5**. Every completed validation
epoch is archived; the lowest-validation-loss epoch is retained as `_BEST`.
The outer loop changes the partition; the inner loop runs all 12 combinations
on that same partition. After the inner loop, the round's winner is copied as
`_BEST_OVERALL` inside that round's directory before the next round starts.
The existing search-root overall winner is also copied after both rounds finish.
The
[SVTRv2 paper, Section 4.1](https://arxiv.org/html/2411.15858v2#S4.SS1) reports
AdamW with learning rate `6.5e-4`, weight decay `0.05`, and batch size `1024` in
a different, multi-GPU scene-text training setup. The grid retains that optimizer
reference point and tests a half-rate and a lower `1e-4` rate, plus weaker `0.02`
weight decay. These are starting hypotheses, not established optima for Gaussian
blur recovery. The paper's data, objective, schedule, and geometry are not this
project's training recipe.

Batches `2` and `4` are conservative, project-specific choices for the 512x64
model and MPS memory constraints, not values claimed by the paper. They do not
guarantee a fit on every MacBook. Edit `param_grid["batch_size"]` to `[1, 2]`
when memory is tight, or test larger batches after measuring memory on an H100
or H200. A larger batch changes optimization, not only throughput.

The grid searches parameters the CTC trainer actually uses. It does not add
teacher distillation, masked-language-model, cosine-loss, or gradient-accumulation
settings from an unrelated objective. Dropout and stochastic depth remain at
their single-run settings unless their existing argument names (`dropout` and
`drop_path`) are added to the dictionary. Unknown keys and invalid values fail
before model training. No scikit-learn, Optuna, or additional dependency is needed.

For H100/H200, select `--device cuda --precision bf16` and review the batch-size
grid for the available memory. The MPS path continues to compute CTC on CPU
with autograd back to the MPS model; CUDA retains its existing BF16 path.

The search loads the manifest once and builds a reproducible plan before
training. Automatic round `r` uses split seed `--seed + r - 1`, giving seeds
`42` and `43` with the current two-round default. Existing group allocation,
duplicate-text checks, and non-empty-partition requirements remain unchanged.
There are at most `SPLIT_ROUNDS` seed attempts, currently two, never more than
five. Duplicate membership is skipped and recorded in `skipped_rounds`;
round numbering retains the original attempt number. Unequal group sizes or
small datasets can therefore produce fewer rounds than requested. Explicit manifest
partitions produce one round without reshuffling.

Within a round, every trial receives the exact same frozen split object and
training probe. Each starts with fresh weights, optimizer, scheduler, and
early-stopping state. The training seed remains the original `--seed` across
all trials and rounds, so this update varies membership rather than deliberately
changing model initialization. The probe changes only as needed for the round's
training membership and stays fixed for that round.

This is repeated holdout, not disjoint-fold cross-validation: held-out samples can
overlap across rounds, and every sample is not guaranteed a turn in the test
partition. No cross-round metric averaging or model ensembling is introduced.

Grid values override matching single-run flags. For example, `--batch-size 1`
does not override the search's `[2, 4]`; edit the dictionary instead. Other flags
apply to every trial. `--epochs` can lower the per-trial limit for smoke tests;
values above 50 are rejected. Changing batch size changes updates per epoch.

Validation CTC loss still selects the best epoch within each trial. That
checkpoint is reloaded and evaluated on the full test partition once. The
completed trials are ranked by **lowest test CER**, with exact ties resolved
by lower test WER, higher test exact match, lower test CTC loss, then ascending
trial path. Paths sort by split-round number, then combination number, so a
complete tie retains the earlier round/trial. The ranking order is fixed and
recorded in `grid_search.json`.

After each completed trial, its scores are compared with both the current
round's winner and the search-wide running winner. The round winner resets
at the next outer iteration; the search-wide winner does not. `grid_search.json`
immediately records the search-wide `best_trial` (a relative path such as
`split_02/trial_0007`), `best_params`, `best_test`, `best_split_round`, and
`best_split_seed`.

After all configurations in a round finish, the round winner's `_BEST.pth` and
matching `_BEST.json` are copied into `split_XX/` with `_OVERALL` appended before
the extension. `round_summary_split_XX.json` records the winning trial, parameters, test
metrics, and copied filenames. The root `grid_search.json` appends the same
information to `round_results`. These copies and reports are complete before
any model in the next round starts training.

After all scheduled rounds finish, the best trial across both rounds is also
copied to the search root. This preserves the existing search-wide selection.
At either level the resulting suffixes are `_BEST_OVERALL.pth` and
`_BEST_OVERALL.json`. Supporting config, vocabulary, JSON, CSV, and text files
are copied with `_OVERALL`, including the winner's membership files. Latest
epoch files and epoch archives are not copied; all original trial files remain
in their trial directories. No model is retrained after selection.

The root `best_files` remains empty until the entire search completes. Errors
stop the search and are recorded. Ctrl+C preserves epoch archives, trial bests,
and already completed round winners. An unfinished round does not publish its
winner, and an unfinished search does not publish a search-root winner.
Automatic training/search resume is not implemented.

## 6. Validation, learning-rate reduction, and early stopping

| Setting | Default |
|---|---|
| Optimizer | AdamW |
| Initial target learning rate | 0.0003 |
| Warmup | 3 epochs, per-batch linear ramp from 10% toward the target LR |
| Weight decay | 0.01; biases and normalization parameters excluded |
| Dropout / stochastic depth maximum | 0.1 / 0.1 |
| Gradient norm clipping | 1.0 |
| Evaluation | Every epoch, `eval()` and inference mode, FP32 |
| Checkpoint selection | Lowest full-validation CTC loss |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 3, minimum LR 0.000001 |
| Early-stopping patience | 5 non-improving epochs after warmup |
| Minimum improvement | 0.0001 absolute, target-length-normalized loss |
| Minimum / maximum duration | 1 / 50 epochs; warmup still excludes its epochs from patience |

Two loss traces are collected for training:

`train_loss` is the per-sample weighted mean of losses observed while the model
is learning during the epoch. Dropout and training-mode batch normalization are
active, and the parameters change from batch to batch.

`train_eval_loss` is computed after the epoch on a fixed training subset of up
to 256 samples, using the same FP32/evaluation settings as validation. Compare
this trace with `val_loss` when checking for overfitting. Change the subset
size with `--train-eval-samples`.

The validation loss uses all validation samples, without random augmentation.
The underlying CTC loss is divided by each target's length before averaging,
matching the meaning of PyTorch's `CTCLoss(reduction="mean")`. Aggregation
weights partial batches by their number of examples.

The checkpoint policy is deliberately separate from the stopping policy:

1. Each completed validation epoch writes an immutable
   `model_<grid_values>_epoch_0001.pth` / `.json` pair, with the epoch number
   incremented for each snapshot. It also updates the unsuffixed latest pair.
   Any new lowest validation loss updates `model_<grid_values>_BEST.pth` and
   its `.json`, even when the improvement is smaller than `min_delta`.
2. Only a sufficient improvement resets the patience counter. Improvements
   accumulate relative to the last sufficiently improved value.
3. Warmup never consumes patience, and stopping cannot occur before
   `min_epochs` (now 1). After warmup, five consecutive epochs without a
   sufficient improvement end training with the default patience. A new
   sufficient improvement resets the count; five total epochs is not the rule.
4. The saved best checkpoint is reloaded before final predictions and any test
   evaluation. The final, possibly overfit epoch does not replace it.

The stopping monitor compares `train_eval_loss`, not noisy training-mode batch
loss, with validation. It flags `overfitting_suspected` when validation loss is
higher than the best validation loss by more than `min_delta` while training
probe loss is lower than at that best epoch by more than `min_delta`. This flag
is saved in `history_split_XX.csv` (`history.csv` for single runs) and checkpoint metadata. When patience is exhausted,
`early_stopping_reason` records `overfitting_suspected` or
`validation_not_improving` in the run summary and finalized latest/BEST metadata.

Training improvement does not reset validation patience. Stalled or worsening
training also does not disable early stopping. A single fluctuation cannot
exhaust patience 5; the flag is diagnostic, not proof of overfitting. The
validation-best checkpoint, rather than the epoch immediately before stopping,
is the last known good model selected by this policy.

The scheduler monitors validation loss after warmup and can lower the learning
rate before the stopping patience expires. Non-finite losses or gradients
raise an error. `zero_infinity=False` prevents invalid examples from silently
turning into zero training loss.

## 7. Outputs

Every grid trial has its own directory, and **all trial filenames include the
outer split number**. Round 1, configuration 1 has this layout:

```text
runs/gaussian_grid/split_01/trial_0001/
├── model_split_01_<grid_values>_epoch_0001.pth
├── model_split_01_<grid_values>_epoch_0001.json
├── ...                                  # One pair for each completed validation epoch
├── model_split_01_<grid_values>.pth      # Latest completed epoch; FP32 safetensors
├── model_split_01_<grid_values>.json
├── model_split_01_<grid_values>_BEST.pth # Lowest-validation-loss epoch
├── model_split_01_<grid_values>_BEST.json
├── config_split_01.json
├── vocab_split_01.json
├── best_split_01.json                   # Compact selected-epoch/validation summary
├── history_split_01.csv                 # Per-epoch losses, metrics, LR, patience
├── splits_split_01.json                 # Membership and fixed training probe
├── splits_split_01.txt                  # Bucket, sample ID, relative image paths
├── run_config_split_01.json             # Arguments, environment, provenance hashes
├── summary_split_01.json
├── validation_predictions_split_01.csv
└── test_predictions_split_01.csv
```

Round 2 uses `split_02/` and `_split_02` filenames. Each of the 12 configurations
gets a separate `trial_XXXX/` directory within that round. There is no shared
history or predictions file that gets overwritten by the next configuration.

`<grid_values>` contains each searched parameter name and its actual value,
in `param_grid` order. For example, one default-grid trial writes:

```text
model_split_01_learning_rate_0.0001_batch_size_2_weight_decay_0.02_epoch_0001.pth
model_split_01_learning_rate_0.0001_batch_size_2_weight_decay_0.02_epoch_0001.json
model_split_01_learning_rate_0.0001_batch_size_2_weight_decay_0.02_epoch_0002.pth
model_split_01_learning_rate_0.0001_batch_size_2_weight_decay_0.02_epoch_0002.json
model_split_01_learning_rate_0.0001_batch_size_2_weight_decay_0.02.pth
model_split_01_learning_rate_0.0001_batch_size_2_weight_decay_0.02.json
model_split_01_learning_rate_0.0001_batch_size_2_weight_decay_0.02_BEST.pth
model_split_01_learning_rate_0.0001_batch_size_2_weight_decay_0.02_BEST.json
```

Explicit single runs use `model_<grid_values>` without a split number, and
retain `config.json`, `vocab.json`, `best.json`, `history.csv`, `splits.json`,
`splits.txt`, `run_config.json`, `summary.json`, and the unsuffixed prediction CSVs.
These names indicate single-run mode, not the first grid permutation.
Grid models carry the round identifier even when copied outside their trial directory.
Each round has a separate directory, so identical parameter combinations in
different rounds do not overwrite one another. Other supported
parameters added to `param_grid` also appear in grid-trial filenames. Numeric
values are not rounded for naming. Each run keeps every completed epoch pair,
plus one latest pair and one best pair. Old best weights remain in their epoch
archives even after the `_BEST` alias changes. Partially trained/unvalidated
epochs are not archived. Budget disk space for up to 1,200 epoch snapshots with
the two-round, 12-configuration, 50-epoch defaults, plus latest/BEST/OVERALL copies.
The `.pth` extension does **not** change the serialization format: these files
are written and read with `safetensors`, not `torch.save` or `torch.load`.

CER and WER are corpus-level edit-distance ratios, not averages of individual
sample percentages. CER includes case, punctuation, and spaces. WER compares
whitespace-delimited tokens. Either error rate can exceed 1.0 when insertions
are numerous. Exact match is the fraction of fully correct labels.

Within each run, the selected epoch is **best by validation CTC loss**, not
necessarily by CER or exact-match accuracy. Across a completed grid, the winner
is selected by **test CER** and the documented tie-breakers. All recognition
metrics are retained so a falling loss cannot conceal poor recognition.

The checkpoint contains weights and JSON metadata. Optimizer and scheduler states are not saved,
so this version does not provide exact training resume. Ctrl+C preserves the
best completed checkpoint and runs final evaluation when one exists. An
interruption before the first validation pass cannot produce a selected model.

### Checkpoint metadata and grid outputs

Every saved training checkpoint embeds a `training` JSON payload in its
safetensors header and writes the same payload to a same-stem `.json` file.
Metadata version 3 includes `grid_params`, all selected hyperparameters,
architecture dimensions, vocabulary, preprocessing, resolved device/precision,
command-line arguments, source/manifest/split hashes, checkpoint epoch, and
validation metrics, FP32 training-probe metrics, training-mode loss, and
patience/overfitting diagnostics. The embedded `run_config` additionally records `split_round`
(`null` for a single run), `split_seed`, `split_sha256` for the matching
`splits_split_XX.json`, and `split_membership_sha256` for `splits_split_XX.txt`
(or the unsuffixed files in single-run mode). The original training seed remains
in `hyperparameters["seed"]` and `run_config["arguments"]["seed"]`. Checkpoint
metadata and the copied winner can therefore be traced to their precise round
and membership. `checkpoint_selection` is `epoch` for immutable archives,
`latest_epoch` for the unsuffixed snapshot, and `val_loss` for the `_BEST` snapshot.
Archives retain their metadata at save time: `stop_reason: "running"` and that
epoch's completed count, not a later termination status. The finalized latest
and BEST snapshots record the run's actual stop status and completed count.
Only the validation-best snapshot receives final test metrics; neither archives
nor latest-epoch metadata claim those scores.
A selected checkpoint preserved before final evaluation cannot yet contain test
metrics. `best_split_XX.json` (`best.json` for single runs) is the compact
validation summary; the parameter-named
JSON files contain the full metadata. During graceful interruption, finalization
reads the epoch archives' headers rather than trusting in-memory counters,
latest/BEST aliases, or stale JSON sidecars. It repairs missing sidecars, restores
the latest committed epoch, and selects the lowest-validation-loss committed
epoch even when interruption occurred during an alias update. Archive weights
and valid archive sidecars are never rewritten during finalization.

The JSON encoding preserves numeric, boolean, and list types while satisfying
the safetensors requirement for string-valued header metadata. No pickle-based
checkpoint format or fitted scaler is introduced. Read metadata without loading
the model tensors:

```python
import json
from pathlib import Path
from safetensors import safe_open

root = Path("runs/gaussian_grid_mps")
model_path = next(root.glob("model_*_BEST_OVERALL.pth"))
with safe_open(str(model_path), framework="pt", device="cpu") as checkpoint:
    metadata = json.loads(checkpoint.metadata()["training"])

# The companion JSON contains the same metadata, without opening the model file.
assert metadata == json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
print(metadata["grid_params"])
print(metadata["hyperparameters"])
print(metadata["model_config"])
print(metadata["run_config"]["split_round"])
print(metadata["run_config"]["split_seed"])
print(metadata["run_config"]["split_sha256"])
print(metadata["test"])
```

A completed two-round grid has this layout. The search-wide winner is shown
from round 1 as an example; its actual round is retained in every copied filename.

```text
runs/gaussian_grid_mps/
├── splits.json                          # Round index, membership paths/hashes, sizes
├── splits.txt                           # All scheduled rounds; round/seed columns
├── grid_search.json                     # Plan, status, trial scores, running/overall winner
├── grid_results.csv                     # One row per evaluated trial
├── split_01/
│   ├── splits_split_01.json             # Frozen membership and probe for seed 42
│   ├── splits_split_01.txt
│   ├── trial_0001/                      # Full round-numbered trial artifacts
│   ├── ... trial_0012/
│   ├── round_summary_split_01.json
│   ├── model_split_01_<grid_values>_BEST_OVERALL.pth
│   ├── model_split_01_<grid_values>_BEST_OVERALL.json
│   └── ..._split_01_OVERALL.*           # Winning config, history, membership, predictions
├── split_02/                            # Seed 43; its own grid, files, and round winner
├── model_split_01_<grid_values>_BEST_OVERALL.pth
├── model_split_01_<grid_values>_BEST_OVERALL.json
├── config_split_01_OVERALL.json
├── vocab_split_01_OVERALL.json
├── best_split_01_OVERALL.json
├── run_config_split_01_OVERALL.json
├── splits_split_01_OVERALL.json
├── splits_split_01_OVERALL.txt
├── summary_split_01_OVERALL.json
├── history_split_01_OVERALL.csv
├── validation_predictions_split_01_OVERALL.csv
└── test_predictions_split_01_OVERALL.csv
```

There can be fewer round directories when explicit assignments or duplicate
partitions reduce the plan. Root membership files describe the scheduled plan,
not proof that each trial completed; consult `grid_search.json` for execution
status. A completed grid root remains loadable by `predict.py` without changing
the inference command. To load a round winner, use `--model-dir
runs/gaussian_grid_mps/split_01`. To load an individual trial, give its nested directory,
for example `--model-dir runs/gaussian_grid_mps/split_02/trial_0003`.

`grid_search.json` records requested/actual/completed split-round counts,
`trials_per_split`, `total_trials`, the round plan, skipped duplicate rounds, and
`round_results` for completed rounds. Each completed round also retains its own
`round_summary_split_XX.json` before the next round starts.
It updates `best_trial`, `best_params`, `best_test`, `best_split_round`, and
`best_split_seed` after each completed trial. `best_files` lists the final copies
only after all scheduled trials complete.

`grid_results.csv` includes the trial path, round, split seed, split hash,
parameter values, selected epoch, stop status, smoke-test marker, validation
loss, test loss, CER, WER, exact match, and test sample count. CER/WER are
lower-is-better ratios; exact match is a higher-is-better fraction. Scores are
not multiplied by 100 in these files.

### Verify which mode ran

A flat output directory containing parameters `learning_rate=0.0003`,
`batch_size=32`, and `weight_decay=0.01`, without `grid_search.json` or
`split_XX/trial_XXXX/`, matches the old single-run defaults. Those values are
not one of the default grid combinations. The old entry point required
`--grid-search`; it skipped both loops when that flag was absent.

For existing output, inspect `run_config.json` -> `arguments.grid_search` and
`split_round` to confirm the recorded mode. New default runs write
`grid_search.json` with `trials_per_split=12`, `total_trials=24`, and
`completed_split_rounds=2` after successful completion with distinct partitions.
The root `splits.txt` is an aggregate membership plan, not a single round's split.

Existing output is not renamed or resumed. Start the corrected default workflow
in a new directory, or explicitly preserve/move an older directory before reuse.
`predict.py` reads both the new round-numbered bundles and earlier unnumbered
`.pth` and `best*.safetensors` layouts. Matching configuration/vocabulary files
must be present; it does not silently substitute files from another round.

## 8. Load on a MacBook using MPS or CPU

Copy the project code and the trained run directory to the Mac. Inference
requires the architecture implementation plus these three matching grid-trial
artifacts (replace `XX` with the split number):

```text
model_split_XX_<grid_values>_BEST.pth
config_split_XX.json
vocab_split_XX.json
```

An explicit single run retains `model_<grid_values>_BEST.pth`, `config.json`,
and `vocab.json`.

Safetensors stores tensors and optional text metadata, not executable Python
architecture or preprocessing code. Architecture and vocabulary are embedded
as metadata for inspection, but this inference loader still uses matching
configuration/vocabulary sidecars. Keep the three model artifacts together.

For a grid winner, the corresponding names are
`model_split_XX_<grid_values>_BEST_OVERALL.pth`,
`config_split_XX_OVERALL.json`, and `vocab_split_XX_OVERALL.json`.
Point `--model-dir` at the search root or a completed `split_XX/` directory;
`predict.py` detects the corresponding bundle automatically. For a trial or
single run, it selects the corresponding `_BEST.pth` rather than the latest epoch. Keep the
matching parameter-named metadata JSON with the model for inspection, although
inference still reads the configuration and vocabulary sidecars.

Legacy `best.safetensors` / `config.json` / `vocab.json` directories and legacy
`best_BEST.safetensors` / `config_BEST.json` / `vocab_BEST.json` grid directories
remain supported. The loader prioritizes a new overall winner, then a new
trial-best model, then the legacy bundle. Multiple matching selected checkpoints
raise an error rather than choosing a model arbitrarily. A missing configuration
or vocabulary sidecar raises an error; the loader never mixes suffixed and
unsuffixed configurations. No new command-line options are required.

```bash
python predict.py \
  --model-dir runs/gaussian_grid_mps \
  --image data/output/after/00000001.png \
  --device mps
```

MPS prediction:

```bash
python predict.py \
  --model-dir runs/gaussian_svtrv2 \
  --image data/output/after/00000001.png \
  --device mps
```

CPU prediction:

```bash
python predict.py \
  --model-dir runs/gaussian_svtrv2 \
  --image data/output/after/00000001.png \
  --device cpu
```

`--device auto` selects CUDA, then MPS, then CPU according to runtime
availability. Explicit `--device mps` errors when MPS is unavailable rather
than concealing a CPU fallback. If a particular PyTorch/macOS combination
cannot execute an operator on MPS, use `--device cpu`.

Prediction loads weights on CPU first, moves the model to the selected device,
and runs FP32. The same weights work without retraining or a CUDA-to-MPS file
conversion. No CTC loss operation is used during prediction: greedy decoding
collapses consecutive duplicate indices, then removes blanks. A blank between
two equal characters preserves both characters.

## References

- [SVTRv2 paper: training settings and implementation details](https://arxiv.org/html/2411.15858v2#S4.SS1)
- [Test-set reuse and hyperparameter selection](https://scikit-learn.org/stable/modules/cross_validation.html)

- [OpenOCR SVTRv2 documentation](https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md)
- [SVTRv2 CTC-only configuration](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/svtrv2_rctc.yml)
- [SVTRv2LNConvTwo33 encoder](https://github.com/Topdu/OpenOCR/blob/main/openrec/modeling/encoders/svtrv2_lnconv_two33.py)
- [Feature-rearrangement RCTC decoder](https://github.com/Topdu/OpenOCR/blob/main/openrec/modeling/decoders/rctc_decoder.py)
- [PyTorch CTCLoss](https://docs.pytorch.org/docs/2.10/generated/torch.nn.CTCLoss.html)
- [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/2.10/amp.html)
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Safetensors PyTorch API](https://huggingface.co/docs/safetensors/api/torch)
