# DSCI-510-Final-Project: Analysis of Key Factors for Movie Success Using TMDb Data

## Project Overview

This project analyzes factors that contribute to a movie’s commercial and critical success using data from **The Movie Database (TMDb) API**. We explore relationships between **budget, revenue, popularity, ratings, genres, release timing**, and **movie age**, and build simple predictive models to forecast box office performance.

The project emphasizes **reproducibility** by separating data collection, cleaning, analysis, and visualization into modular Python scripts.

---
## Research Questions

1. **Budget vs. Revenue**  
   What is the relationship between a movie’s production budget and its box office revenue? Is a higher budget a reliable predictor of financial success?

2. **Popularity, Ratings, and Success**  
   How are TMDb popularity, user ratings, and box office revenue related?

3. **Genre Analysis & ROI**  
   How do different movie genres differ in typical budget, revenue, and return on investment (ROI)?

4. **Movie Age and Ratings**  
   How does a film’s age correlate with its audience rating?

5. **Forecasting (Extended Analysis)**  
   To what extent can box office revenue be explained using features such as budget, popularity, ratings, and release timing through a simple predictive model?

---

## Data Source

* **TMDb API** (The Movie Database)
* Movies released from **1990 onward**
* Data includes:

  * Movie metadata (budget, revenue, release date, popularity, ratings)
  * Genre information
  * Director and top cast (where available)

> Note: An API key is required to run `get_data.py`.

## TMDb API Key Setup

This project uses the TMDb API, which requires a personal API key.

Please obtain a free API key from:
https://www.themoviedb.org/documentation/api

Set the API key as an environment variable before running `get_data.py`.

### macOS / Linux
```bash
export TMDB_API_KEY="your_api_key_here"
```

---

## Project Structure

```
.
├── src/
│   ├── get_data.py              # Fetch raw data from TMDb API
│   ├── clean_data.py            # Clean and transform raw data
│   ├── run_analysis.py          # Statistical analysis & modeling
│   └── visualize_results.py     # Generate and save visualizations
│
├── data/
│   ├── raw/                     # Raw JSON data from API
│   └── processed/               # Cleaned CSV + analysis outputs
│
├── results/
│   └── figures/                 # Saved visualization images (.png)
│
├── visualization.ipynb      # Notebook for presenting visualizations
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Fetch data from TMDb

```bash
python src/get_data.py
```

### 3. Clean and preprocess data

```bash
python src/clean_data.py
```

### 4. Run analysis and models

```bash
python src/run_analysis.py
```

### 5. Generate visualizations

```bash
python src/visualize_results.py
```

All figures will be saved to:

```
results/figures/
```

---

## Notebooks

* **`visualization.ipynb`** is provided **only for visualization and presentation**.
* Core logic and computations reside in `.py` files for reproducibility.
* The project can be fully reproduced **without notebooks**.

---

## Outputs

* **Clean dataset:**
  `data/processed/movies_clean.csv`
* **Analysis results:**
  `data/processed/analysis_results/`
* **Figures:**
  `results/figures/*.png`

These figures are used directly in the final report.

---

## Key Libraries

* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn`
* `requests`

---

## Team Members

* **Yuchen Wu**
  Email: [ywu55711@usc.edu](mailto:ywu55711@usc.edu)
  USC ID: 9788517696
  GitHub: ywu55711

* **Yang Zhao**
  Email: [yzhao657@usc.edu](mailto:yzhao657@usc.edu)
  USC ID: 7748154612
  GitHub: yzhao657-lyz
