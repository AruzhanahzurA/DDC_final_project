# Digital Phenotyping with Context-Aware Confound Control

Authors: Aruzhan Oshakbayeva, Azhar Serik, Libby Thogmartin

Course: Digital Data Collection Methods

## Project Overview

This project investigates whether contextual confound variables change the
relationship between passive smartphone signals and self-reported mood. We
build a responsible, bias-aware data collection pipeline using the StudentLife
dataset (Dartmouth College, Wang et al. 2014) — a publicly available dataset
of passive sensor data and ecological momentary assessments (EMA) collected
from 48 undergraduate students over a 10-week spring term.

## Research Question

If we combine passive smartphone data with contextual confound data, does the
relationship between phone signals and self-reported mood change?

## Dataset

StudentLife (Dartmouth, 2013)
Download: https://studentlife.cs.dartmouth.edu/dataset.html

Place the downloaded dataset in the `data/studentlife/` folder before running
any notebooks. This folder is gitignored and will not be pushed to GitHub.

## Setup Instructions

1. Clone the repository:

```bash
git clone <repo-url>
cd DDC_final_project
```

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate    # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place the StudentLife dataset under `data/studentlife/` (the repository
    expects the original StudentLife folder structure under this path).

5. Start Jupyter and open the main notebook:

```bash
jupyter notebook code/Final_project.ipynb
```

6. Run cells in order. If you only have `data/processed/` CSVs, the notebook
    will skip expensive external fetches and load processed files directly.

Repository Structure (expanded)

- `data/studentlife/` — original StudentLife raw files, including `EMA/`,
   `education/`, and `sensing/`.
- `data/processed/` — cleaned, per-participant/day aggregated CSVs produced
   by the notebook.
- `code/Final_project.ipynb` — main analysis notebook (preprocessing,
   feature engineering, modeling, plotting).
- `output/` — figures and tables produced by the notebook; check it after
   running the analysis.

## Repository tree
---------------
```
.
├─ README.md
├─ requirements.txt
├─ LICENSE
├─ .gitignore
├─ mood_daily1.csv
├─ data/
│  ├─ processed/
│  │  ├─ activity.csv
│  │  ├─ conversation.csv
│  │  ├─ dark.csv
│  │  ├─ deadlines.csv
│  │  ├─ gps.csv
│  │  ├─ pam.csv
│  │  └─ weather.csv
│  └─ studentlife/
│     ├─ EMA/
│     ├─ education/
│     │  └─ deadlines.csv
│     └─ sensing/
│        ├─ activity/
│        │  ├─ activity_u00.csv
│        │  ├─ activity_u01.csv
│        │  └─ ...
│        ├─ conversation/
│        │  ├─ conversation_u00.csv
│        │  └─ ...
│        └─ gps/
│           └─ ...
├─ code/
│  └─ Final_project.ipynb
└─ output/
```

## Detailed Architectural & Preprocessing Pipeline

The analysis follows a staged architecture:

- Ingestion: raw sensor CSVs and EMA JSON files are read from `data/sensing/`
   and `data/EMA/` respectively. External sources are fetched optionally via
   HTTP APIs and can be cached locally:
   - **Open-Meteo** provides historical weather variables used to derive daily
     weather categories.
   - **GDELT** supplies news-event sentiment and volume signals as a societal
     mood proxy.
   - **Google Trends** supplies search interest signals for anxiety-related terms.
   These extra-context signals are used to build a richer confound-aware model.
- Privacy-preserving mapping: raw participant UIDs are irreversibly hashed
   (SHA-256 with a salt) immediately upon ingestion; raw UIDs and raw GPS
   coordinates are not stored in processed outputs.
- Sensor-specific filtering & aggregation:
   - Activity: compute `frac_stationary` per day (fraction of stationary
      readings) and total readings count.
   - Dark periods: sum screen-off durations >= 1 hour to approximate sleep
      (`dark_hours`).
   - Conversation: sum detected conversation durations to produce
      `conversation_hours` (no audio content is stored).
   - GPS: filter to high-accuracy fixes (<=100 m), compute per-day median
      coordinate, then derive `km_from_campus` with the Haversine formula and
      discard raw coordinates.
   - Deadlines: reshape education/deadlines data to per-participant/day
      counts and derive `has_deadline` flags.
- Time handling: timestamps are parsed, converted to a common timezone,
   normalized to dates, and the study period is enforced (Spring 2013).
- Merge: outer-merge all per-sensor daily tables on `hashed_uid` + `date`,
   then drop rows missing the primary outcome (daily mean PAM score).
- Feature engineering: add weekday/weekend and holiday flags, derive
   `weather_category` from weather codes, and **construct composite societal mood signals** from GDELT + Google Trends.
- Save: processed CSVs are written to `data/processed/` for reproducibility.

## Limitations & Future Work

- Limitations:
   - Sample: StudentLife is a small, convenience sample (48 participants)
      from a single institution and term — limits generalizability.
   - Missingness & measurement error: sensor dropouts, variable EMA
      response rates, and GPS inaccuracies may bias results.
   - Confounding: unobserved confounders may remain; observational analysis
      cannot fully establish causality.
   - Temporal scope: the 10-week Spring 2013 window may not capture seasonal
      or multi-term patterns.

- Future work:
   - Apply causal methods (instrumental variables, difference-in-differences,
      or causal graphs) or mixed-effects models to account for within-person
      clustering.
   - Improve missing-data handling with principled imputation and sensitivity
      analysis.
   - Expand validation to other datasets and longer timeframes.
   - Automate and cache external API fetches and add retry/backoff logic.
   - Add unit tests and CI to ensure reproducible preprocessing and analysis.

Setup notes

- If you already have `data/processed/` CSVs, you can skip raw-sensor
   preprocessing and run the analysis directly.
- Replace placeholder contact information in this README before public
   release if desired.

Citation

Wang, R., Chen, F., Chen, Z., Li, T., Harari, G., Tignor, S., Zhou, X.,
Ben-Zeev, D., & Campbell, A. T. (2014). StudentLife: Assessing Mental Health,
Academic Performance and Behavioral Trends of College Students using
Smartphones. UbiComp 2014.
   the notebook.

Data provenance & notes
-----------------------
- Source: StudentLife dataset (Dartmouth) — see the original dataset page for
   collection details and usage terms.
- Preprocessing: documented inline in the notebook. Key preprocessing steps
   include resampling, aggregation, and the creation of confound variables.
- Limitations: be mindful of class imbalance, missingness, and sensor
   heterogeneity across participants.

License & Citation
------------------
This project is provided under the terms in [LICENSE](LICENSE). If you use
this code or data in a publication, please cite the StudentLife paper and the
project authors:

Wang, R., Chen, F., Chen, Z., Li, T., Harari, G., Tignor, S., Zhou, X.,
Ben-Zeev, D., & Campbell, A. T. (2014). StudentLife: Assessing Mental Health,
Academic Performance and Behavioral Trends of College Students using
Smartphones. UbiComp 2014.


Acknowledgements
----------------
Thanks to the StudentLife team for publicly sharing the dataset and to course
mentors for guidance.
