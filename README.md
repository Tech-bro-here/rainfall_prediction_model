# Rainfall Prediction Model

This repository contains a machine learning model for predicting rainfall based on weather data. The model is built using Python and scikit-learn's Random Forest Classifier.

## Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA) with visualizations
- Handling class imbalance through downsampling
- Hyperparameter tuning using GridSearchCV
- Model evaluation metrics
- Serialized model for easy deployment

## Requirements

To run this project, you'll need:

- Python 3.x
- Required Python packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - pickle (built-in)

Install the required packages using:
```
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Clone this repository
2. Ensure you have the `Rainfall.csv` dataset in the same directory
3. Run the `app.py` script:
   ```
   python app.py
   ```

The script will:
- Load and preprocess the data
- Train the Random Forest model
- Evaluate the model performance
- Save the trained model to `rainfall_rf_model.pkl`

## Model Details

- **Algorithm**: Random Forest Classifier
- **Hyperparameters**: Optimized using GridSearchCV
- **Evaluation Metrics**:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)

## File Structure

- `app.py`: Main Python script containing all the code
- `Rainfall.csv`: Input dataset (not included in repository)
- `rainfall_rf_model.pkl`: Serialized trained model (created after running the script)

## Example Prediction

The script includes an example prediction using sample input data:
```python
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
```
The model will output either "Rainfall" or "No Rainfall" based on this input.

## License

This project is open-source and available under the [MIT License](LICENSE).
