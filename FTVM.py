import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.diagnostic import het_breuschpagan

from scipy import stats
# 1. Data Loading and Initial Cleaning (IQR Filters)
df = pd.read_csv('male_players_finale.csv', encoding='utf-8', sep=",")
df = df.sort_values(by=['player_id', 'fifa_version'], ascending=[True, False])
df = df.drop_duplicates(subset=['player_id'], keep='first')

# Value and Wage Filters (Bilateral Interquartile Range)
# We apply this to remove extreme market outliers (like Mbappé or 0-value errors)


# Feature Engineering: Age squared to capture the non-linear decay of value over time
df['age_sq'] = df['age'] ** 2
# Shuffle the dataset to prevent any systematic ordering bias
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. ECONOMETRIC STEP: Residual Outlier Removal (3-Sigma Rule)
# We define temporary features to detect "veterans" or anomalies where the model fails
features = ['overall', 'age_sq', 'wage_eur']
X_temp = df[features]
y_temp_log = np.log1p(df['value_eur'])

# Fit with Statsmodels to perform diagnostic analysis
X_with_const = sm.add_constant(X_temp)  # Manual constant required for Statsmodels
diagnostic_model = sm.OLS(y_temp_log, X_with_const).fit(cov_type='HC3')

# Calculate residuals and filter the original DataFrame
# We remove observations where the error is > 3 standard deviations (the Q-Q plot "tail")
df['residuals'] = y_temp_log - diagnostic_model.predict(X_with_const)
res_std = df['residuals'].std()

df_final = df[df['residuals'].abs() <= 3 * res_std].copy()

print(f"Final Cleaning: Removed {len(df) - len(df_final)} observations with extreme residuals.")

X_final = df_final[features]
y_final_log = np.log1p(df_final['value_eur'])

X_final_const = sm.add_constant(X_final)
model_final = sm.OLS(y_final_log, X_final_const).fit(cov_type='HC3')

# Get new residuals to make the following tests:
residuals_final = model_final.resid

print("\n=== FINAL MODEL (WITHOUT RESIDUAL OUTLIERS) ===")
print(model_final.summary())

# VIF

X_final_vif = X_final
vif_final = pd.DataFrame()
vif_final["Feature"] = features
vif_final["VIF"] = [variance_inflation_factor(X_final_vif.values, i) for i in range(len(features))]
print(vif_final)

#Autocorrelation test
bg_test = acorr_breusch_godfrey(model_final, nlags=1)

print(f"Breusch-Godfrey Lagrange Multiplier statistic: {bg_test[0]:.4f}")
print(f"Breusch-Godfrey p-value: {bg_test[1]:.4f}")

#Heteroscedasticity test
bp_final = het_breuschpagan(residuals_final, model_final.model.exog)
print(f"--- 3. p-value Breusch-Pagan (Final): {bp_final[1]}")

# 3. MACHINE LEARNING STEP: Final Training
# We use df_final, which is now free of noise and structural market anomalies

# Split the clean data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final_log, test_size=0.2, random_state=42)

# Scikit-Learn Linear Regression (Handles the intercept automatically)
model = LinearRegression()
model.fit(X_train, y_train)

# Final performance metric
# r2_test = model.score(X_test, y_test)
# print(f"Model trained successfully. Test R2: {r2_test:.4f}")

for i, name in enumerate(X_final):
    print(f'{name:>10}: {model.coef_[i]}')
# Check accuracy on the test set
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
print(f"Model Accuracy (R2 Score): {r2:.2f} (Closer to 1.0 is better)")

# IMPORTANT: Make predictions for the WHOLE dataset
# This adds a new column 'predicted_value' next to the actual 'value_eur'

log_preds = model.predict(X_final)
df_final['predicted_value'] = np.expm1(log_preds)
df_final['predicted_value'] = df_final['predicted_value'].round(2)

# Let's see the difference for the first 5 players
print(df_final[['short_name', 'value_eur', 'predicted_value']].head())

# Translate y_test back to real money
y_test_real = np.expm1(y_test)
# Translate your test predictions back to real money
predictions_test_real = np.expm1(model.predict(X_test))

mae_euros = mean_absolute_error(y_test_real, predictions_test_real)
print(f"On average, the model is off by: €{mae_euros:,.2f}")

mape = np.mean(np.abs((y_test_real - predictions_test_real) / y_test_real)) * 100
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# Save to a new CSV
# This 'clips' the values so 0 is the absolute minimum
df_final['predicted_value'] = df_final['predicted_value'].clip(lower=0)
df_final['model_r2'] = float(r2)
def age_category(age):
    if age <= 21: return '1. Young Prospect (Under 21)'
    elif age <= 25: return '2. Emerging (22-25)'
    elif age <= 29: return '3. Prime(26-29)'
    elif age <= 34: return '4. Veteran(30-34)'
    else: return '5. Late Career(+35)'# Apply it to your dataframe

df_final['age_group'] = df_final['age'].apply(age_category)
df_final.to_csv('male_players_with_predictions_test.csv', index=False, encoding='utf-8-sig', sep=',', decimal='.')
print("File saved! Ready for Power BI.")