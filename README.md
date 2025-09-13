# DEX Valuation Study

This repository contains code and data for a study on predicting decentralized exchange (DEX) token valuations using fundamental metrics.  The project is organised into several modules:

- **data/**: Raw and processed data. Raw API responses should be stored in `data/raw/` and processed, cleaned datasets in `data/processed/`.
- **src/**: Core source code for data collection, preprocessing, model training, and evaluation.
- **notebooks/**: Jupyter notebooks used for exploration and analysis.
- **results/**: Figures and tables produced by the analysis.

## Usage

1. Install the required packages:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Populate the `.env` file with your API keys and RPC URLs. See `.env.example` for guidance.

3. Run the data collection script to fetch daily metrics for each protocol. Refer to the functions in `src/data_collection.py` and update them with real API requests.

4. Validate and preprocess the data using `src/validation.py` and `src/preprocessing.py`.

5. Train and evaluate models using the scripts provided in `src/models.py` and `src/evaluation.py` or explore the data with notebooks in `notebooks/`.

## Documentation

The data dictionary is provided in `data/data_dictionary.md`. Further documentation can be found within individual source files.
