# Pusula_Seyma_Nur_Cin

**Author:** Seyma Nur Cin  
**Email:** cinseymanur@gmail.com

## Project Overview
This project performs **Exploratory Data Analysis (EDA)** and **data preprocessing** on a Physical Medicine & Rehabilitation dataset. The dataset has 2235 observations and 13 features, with the target variable being `TedaviSuresi` (treatment duration in sessions). The goal is to make the data clean, consistent, and ready for predictive modeling.

## Dataset Columns
- `HastaNo` : Anonymized patient ID  
- `Yas` : Age  
- `Cinsiyet` : Gender  
- `KanGrubu` : Blood type  
- `Uyruk` : Nationality  
- `KronikHastalik` : Chronic conditions (comma-separated)  
- `Bolum` : Department/Clinic  
- `Alerji` : Allergies (single or comma-separated)  
- `Tanilar` : Diagnoses  
- `TedaviAdi` : Treatment name  
- `TedaviSuresi` : Treatment duration (target)  
- `UygulamaYerleri` : Application sites  
- `UygulamaSuresi` : Application duration  

## How to Run the Code
The project code is written as a **Python script** (`main.py`) and can be run directly in PyCharm or any Python IDE.

1. Clone this repository:
```bash
git clone https://github.com/seymanrcn/Pusula_Seyma_Nur_Cin.git

Make sure you have the required packages installed:
pip install pandas numpy matplotlib seaborn scikit-learn rapidfuzz openpyxl

Run the main script:
python main.py

This will:

Load the dataset
Perform EDA (visualizations and summaries)
Clean and preprocess the data (handle missing values, encode categorical variables, normalize numeric features)
Save the cleaned dataset if needed

Project Structure
Pusula_Seyma_Nur_Cin/
│
├─ data/
│   └─ data_set.xlsx          # Original dataset
│
├─ main.py                    # Python script with EDA and preprocessing
├─ README.md                  # This file
