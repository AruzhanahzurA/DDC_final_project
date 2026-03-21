# Digital Phenotyping with Context-Aware Confound Control

**Authors:** Aruzhan Oshakbayeva, Azhar Serik, Libby Thogmartin
**Course:** Digital Data Collection Methods

## Project Overview

This project investigates whether contextual confound variables change the
relationship between passive smartphone signals and self-reported mood.
We build a responsible, bias-aware data collection pipeline using the
StudentLife dataset (Dartmouth College, Wang et al. 2014) — a publicly
available dataset of passive sensor data and ecological momentary assessments
(EMA) collected from 48 undergraduate students over a 10-week spring term.

## Research Question

If we combine passive smartphone data with contextual confound data, does
the relationship between phone signals and self-reported mood change?

## Dataset

StudentLife (Dartmouth, 2013)
Download: https://studentlife.cs.dartmouth.edu/dataset.html

Place the downloaded dataset in the data/studentlife/ folder before running
any notebooks. This folder is gitignored and will not be pushed to GitHub.

## Repository Structure

data/                        # raw data, gitignored
code/                        # notebooks go here
docs/                        # documentation
.gitignore
requirements.txt
README.md

## Setup Instructions

1. Clone the repository
2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate        # Mac/Linux
   venv\Scripts\activate           # Windows
3. Install dependencies:
   pip install -r requirements.txt
4. Download the StudentLife dataset and place it in data/studentlife/
5. Open the notebooks in the code/ folder in order

## Citation

Wang, R., Chen, F., Chen, Z., Li, T., Harari, G., Tignor, S., Zhou, X.,
Ben-Zeev, D., & Campbell, A. T. (2014). StudentLife: Assessing Mental Health,
Academic Performance and Behavioral Trends of College Students using
Smartphones. UbiComp 2014.