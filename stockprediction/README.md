# **Stock Price Prediction Using S&P 500 Dataset**

## **Problem Definition**
The objective of this project is to predict whether the S&P 500 stock index price will increase on the following day. This involves leveraging historical stock data to build a classification model that identifies market trends and makes informed predictions to support investment decision-making.

---

## **Data Processing**
### Dataset Overview
- **Source**: S&P 500 historical data.
- **Features**:
  - `Open`: Opening price of the stock index on a given day.
  - `High`: Highest price of the stock index on a given day.
  - `Low`: Lowest price of the stock index on a given day.
  - `Close`: Closing price of the stock index on a given day.
  - `Volume`: Number of shares traded on a given day.
  - `Next Day`: Closing price of the stock index on the next trading day (target variable for derived features).
  - `Target`: Binary variable indicating whether the stock price increased the next day (`1` for increase, `0` for decrease).
  - `Close_Ratio_X`: Ratio of the closing price to the rolling average of the closing price over X days (e.g., `Close_Ratio_2` for 2 days).
  - `Trend_X`: Cumulative sum of the target variable over the past X days (e.g., `Trend_2` for 2 days).

### Preprocessing Steps
1. Cleaned the dataset by handling missing values and unnecessary columns.
2. Added derived features:
   - `Close_Ratio`: Rolling averages of the closing price over different horizons (e.g., 2 days, 5 days, etc.).
   - `Trend`: Cumulative trends over multiple time horizons.

---

## **Model Building**
### Approach
1. **Feature Engineering**:
   - Created additional features to represent financial trends and ratios.
2. **Algorithm**:
   - **Random Forest Classifier** was chosen for its robustness and ability to handle complex feature interactions.
3. **Hyperparameter Tuning**:
   - Adjusted parameters like `n_estimators` and `min_samples_split` to optimize model performance.
4. **Validation**:
   - Implemented a backtesting strategy to evaluate predictions on unseen data.

---

## **Business Insights**
### Feature Importance
- **Short-term Trends**:
  - Features like `Close_Ratio_2` showed a quick reflection of price movement.
- **Long-term Trends**:
  - Trends over 250-day (`Trend_250`) and 1000-day (`Trend_1000`) horizons exhibited strong predictive power for overall market behavior.

### Prediction Performance
- **Precision Score**: Achieved a precision score of **0.54**, indicating moderate predictive capability.
- **Prediction Distribution**:
  - Predicted 1 (Price Increase): ~16%.
  - Predicted 0 (Price Decrease): ~84%.

These insights suggest the potential to identify key market trends and assist in making data-driven investment decisions.
