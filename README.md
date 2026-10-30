# AutoValue AI

AutoValue AI is a Flask web application that estimates used-car selling prices with a trained scikit-learn regression model. The model is trained on the attached Indian used-car dataset (`india_used_cars.csv`), copied into `data/used_cars.csv` without modification.

## Features

- Dataset-driven prediction form with dependent brand, model, and variant dropdowns
- Saved preprocessing pipeline plus selected regression model
- Price estimates in Indian rupees (₹)
- Model insights dashboard built from the real CSV and evaluation metrics
- Input validation on both the browser and the Flask API
- Optional SQLite log of prediction requests

## Technologies

- Frontend: HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js, Lucide icons
- Backend: Python, Flask
- ML: pandas, NumPy, scikit-learn, Joblib

## Dataset

| Item | Value |
| --- | --- |
| Source file | `C:\Users\Abhinav Kumar\Downloads\india_used_cars.csv` |
| Project copy | `data/used_cars.csv` |
| Records | 60,000 |
| Missing values | None |
| Duplicate rows | None |
| Target | `selling_price` (INR) |
| Features | `brand`, `model`, `variant`, `manufacturing_year`, `km_driven`, `fuel_type`, `transmission`, `previous_owners`, `engine_cc`, `mileage_kmpl`, `condition`, `city` |

Currency is treated as INR because listing prices are in rupee-scale integers and cities are Indian metros.

### Limitations

- The file includes manufacturing years through 2025. Those rows were kept as provided.
- Selling prices range from ₹25,000 to ₹1.94 crore. Extreme luxury listings remain in the training data and can increase residual error.
- The CSV does not include accident history, insurance, or service records.

## Preprocessing

1. Standardize categorical text and aliases for fuel, transmission, and condition.
2. Coerce numeric columns and drop invalid non-positive prices or distances.
3. Fit a `ColumnTransformer` on the training split only: `StandardScaler` for numeric fields and `OneHotEncoder(handle_unknown="ignore")` for categoricals.

## Machine learning

Algorithms compared on an 80/20 held-out test split (`random_state=42`):

- Linear Regression
- Random Forest Regressor
- Histogram-based Gradient Boosting (`HistGradientBoostingRegressor`)

**Actual test metrics will be written here after `python train_model.py` completes.** Until then, do not treat any accuracy figure as real.

## Project structure

```text
Car Price Predictor/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/used_cars.csv
├── models/
├── src/
├── templates/
├── static/
└── tests/
```

## Windows setup

```powershell
cd "C:\Users\Abhinav Kumar\Desktop\Git Projects\Car Price Predictor"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

If you need to restore the dataset, copy the original CSV to `data/used_cars.csv`. Do not overwrite the file in Downloads.

## Tests

```powershell
pytest -q
```

## Disclaimer

AutoValue AI produces statistical estimates from historical listings. It is not an official valuation and does not guarantee a selling price.
