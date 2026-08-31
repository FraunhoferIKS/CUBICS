# CUBICS: Situation-aware performance estimation for safety-relevant ML components
---
CUBICS is a runtime performance estimation framework that uses **Subjective Logic (SL)** to represent uncertainty in system guarantees across operational contexts. Rather than treating a system's reliability as a single scalar, CUBICS maintains per-situation opinions (belief, disbelief, uncertainty) that are updated incrementally from observed evidence. A multinomial deduction step then derives a system-wide *marginal guarantee* that properly accounts for how frequently each operational situation is encountered.

The framework is evaluated on object detection (YOLOv12 on BDD100K) under varying weather and time-of-day conditions.


## TARGET BADGE 
---
Reproducible Badge 

## INFO 
---
This repository contains the code to reproduce the results of the paper:

> **CUBICS: Situation-aware performance estimation for safety-relevant ML components**

Submission number: 160
Contact: Benjamin Herd (benjamin.herd@iks.fraunhofer.de), Jessica Kelly (jessica.kelly@fraunhofer.de), Mario Trapp (mario.trapp@tum.de)


## EXPECTED BEHAVIOUR
---
Running the artifact reproduces the per-situation and marginal guarantee estimates
reported in the paper, together with the figures derived from them. No paper result
requires retraining: all three RQ scripts consume the pre-computed
`data/test_metrics_complete.csv` shipped with the repository.

- **`run_rq1.py`** — synthetic Rain × Wind × Time scenario. Prints the marginal
  guarantee at each evidence milestone, paper results are replicated with the
  default `--num-runs 5000 --seed 0`. Writes `situational_progression.png` and
  `marginal_vs_agnostic.png` in the outputs folder.
- **`run_rq2.py`** — Prints per-situation TP/FN and success rate,
  the context opinion ω_S with its situation frequencies, and a comparison table of
  belief / disbelief / uncertainty for the baseline and representative situations (found in Table I of the paper). The results in Table II are obtained after running the script for the person, car, truck, and bike classes. This script also writes the `rq2_*` figures.
- **`run_rq3.py`** — prior sensitivity (Table III) sensitivity to data scarcity, prior choice, label noise, and
  context noise. The context-noise sweep should show the marginal guarantee
  degrading ⟨monotonically⟩ as context-labelling error increases. This script writes the
  `rq3_*` figures.

**Optional retraining.** `train.py` and `evaluate.py` regenerate the CSV from
scratch. This is not needed for any paper result and requires a GPU and a local copy
of BDD100K. 
 

## ARTIFACT DESCRIPTION
---

CUBICs is structured as follows: 

```
cubics/
├── src/cubics/              # Installable Python package — core SL assurance machinery
│   ├── assurance/           #   Context, Conditional, Contract, simulation engine
│   ├── data/                #   CSV loaders and data schemas
│   └── config.py            #   CubicsConfig dataclass and YAML loader
│
├── experiments/             # Experiment scripts 
│   ├── train.py             #   Train YOLOv12 on BDD100K
│   ├── evaluate.py          #   Evaluate model across weather × time scenarios
│   ├── run_rq1.py           #   RQ1: CUBICS simulation (Rain × Wind × Time)
│   ├── run_rq2.py           #   RQ2: Marginal vs agnostic guarantee analysis
│   └── run_rq3.py           #   RQ3: Robustness and sensitivity analysis
│
├── data/                    # Dataset inputs and pre-computed results
│   ├── bdd_100k.yaml        #   BDD100K dataset config (for training)
│   ├── bdd_metadata.csv     #   Per-image weather/time-of-day metadata
│   └── test_metrics_complete.csv   # Pre-computed per-scenario TP/FN
│      
│
├── models/
│   └── best.pt              # Trained YOLOv12-L weights
│
├── outputs/                 # Generated figures (written here by experiment scripts)
```

## ENVIRONMENT SETUP
---

### Requirements 
- Python ≥ 3.10


## GETTING STARTED 
---

### Installation

After cloning the repository, proceed through the following steps to set-up the cubics framework. 

#### 1. Install the required packages 

```bash
pip install -r requirements.txt
```

#### 2. Install the `cubics` package

```bash
cd cubics/
pip install -e .
```

#### 3. Verify

```bash
python -c "from cubics import Context, Contract, run_simulation; print('OK')"
``` 

### Quick Results 

The core SL assurance machinery is importable as a standalone library:

```python
from cubics import Context, Conditional, Contract, run_simulation, make_default_scenario

# Build a context over operational dimensions
context, conditional, probs, fail_rates = make_default_scenario()

# Run the simulation
result = run_simulation(context, conditional, probs, fail_rates, num_runs=5000, seed=0)

# Inspect per-milestone marginal guarantees
for milestone, opinion in zip(result.milestones, result.marginal_history):
    print(f"After {milestone} iterations: E[guarantee] = {opinion.prob()[0]:.4f}")
```

## REPRODUCIBILITY
---

The paper results can be reproduced using the per-scenario TP/FN CSV located in data/test_metrics_complete.csv. However, to fully reproduce the results, the code for training + evaluating YOLOv12 on BDD100K is also provided. 

### Training and Evaluating YOLO on BDD100K 

If you want to retrain from scratch or re-run evaluation against the BDD100K dataset:

#### Step 1 — Download BDD100K

Download the BDD100K dataset and update `data/bdd_100k.yaml` to point to your local copy:

```yaml
path: /path/to/bdd100k
train: images/100k/train
val:   images/100k/val
```

#### Step 2 — Train

```bash
python experiments/train.py --epochs 100 --imgsz 1024 --batch 4 --device 0
```

Copy best weights to `models/best.pt` to use in evaluation.

#### Step 3 — Evaluate

```bash
python experiments/evaluate.py \
    --model models/best.pt \
    --metadata-csv data/bdd_metadata.csv \
    --output-csv data/test_metrics_complete.csv
```

This regenerates the per-scenario TP/FN CSV consumed by the RQ2 and RQ3 scripts.


### Running experiments 

Each script is self-contained:

```bash
# RQ1: CUBICS simulation (Rain × Wind × Time, 8 situations)
python experiments/run_rq1.py --num-runs 5000 --seed 0

# RQ2: Marginal vs agnostic guarantee on real BDD100K evaluation data
python experiments/run_rq2.py --class-name person
python experiments/run_rq2.py --class-name truck
python experiments/run_rq2.py --class-name bike
python experiments/run_rq2.py --class-name car

# RQ3: Robustness to data scarcity, prior choice, label noise, and context noise
python experiments/run_rq3.py
```

All scripts default to reading from `data/` and writing to `outputs/`. 

Figures are written to `outputs/`:

| File | Figure in paper |
|---|---|
| `outputs/situational_progression.png` | Fig. 3 — Progression of beta distributions across situational guarantees. |
| `outputs/marginal_vs_agnostic.png` | - |
| `outputs/rq2_marginal_vs_agnostic.png` | Fig. 5 — SL marginal vs agnostic (RQ2) |
| `outputs/rq2_situational_guarantees.png` | - |
| `outputs/rq2_scenario_overlays.png` | - |
| `outputs/rq2_combined_analysis.svg` | Fig. 4 — Beta PDFs for the person class |
| `outputs/rq3_prior_sensitivity.svg` | - |
| `outputs/rq3_context_sensitivity_curve.svg` | Fig. 6 — context sensitivity curve |

---

## CITATION
---

If you use this code, please cite:

```bibtex
@inproceedings{cubics2025,
  title     = {[CUBICS: Situation-aware performance estimation for safety-relevant ML components]},
  author    = {[Benjamin Herd, Jessica Kelly, Mario Trapp]},
  booktitle = {[ISSRE]},
  year      = {2026}
}
```
