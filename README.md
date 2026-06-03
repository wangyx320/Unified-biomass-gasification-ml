# Unified-biomass-gasification-ml

This repository provides code, data, and trained models for a unified temperature-driven machine learning framework for multi-configuration fixed-bed biomass gasification.

## Features

- **Data Processing:** Load experimental gasification data from a CSV file for model training and prediction.
- **Multiple Machine Learning Models:** Includes Decision Tree, Support Vector Machine, Multilayer Perceptron, Random Forest, and XGBoost models.
- **Pre-trained Models:** Provides trained model files for reproducing prediction results.
- **Prediction Outputs:** Predicts syngas composition and lower heating value, including H2, CO, CH4, CO2, and LHV.
- **Reproducibility:** Includes dataset, source code, trained models, and environment requirements associated with the manuscript.

## Requirements

- Python 3.x

  Required Python packages are listed in `requirements.txt`.
  Install the required packages using:
```bash
  pip install -r requirements.txt
```

## Usage

1. Download or clone this repository.
2. Install the required packages using `requirements.txt`.
3. Run the corresponding Python script for each model.
4. The dataset is stored in the `Resource/` folder.
5. Check and adjust the dataset path in each script if necessary.

## File Structure

- **Resource/:** Contains the experimental dataset.
- **Model/:** Contains the model scripts and trained model files.
- **requirements.txt:** Lists the required Python packages.
- **README.md:** Project description and usage instructions.
- **LICENSE:** License information.

## Note

Due to data-sharing restrictions, only a partial dataset is publicly provided in this repository. The full dataset may be made available from the corresponding author upon reasonable request and subject to applicable data-use restrictions.

The dataset, code, and trained models are intended for academic and research use.
