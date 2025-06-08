# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 12:46:35 2025

@author: there
"""

import json
import pandas as pd
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
#MODEL CREATION
'''
# Load the JSON file
with open('public_cases.json', 'r') as f:
    data = json.load(f)


flattened_data = [{**entry['input'], 'expected_output': entry['expected_output']}
    for entry in data]

# Create a DataFrame
df = pd.DataFrame(flattened_data)


df["receipts_per_day"] = df["total_receipts_amount"] / df["trip_duration_days"]
df["receipts_per_mile"] = df["total_receipts_amount"] / df["miles_traveled"]


# Sweet spot trip length 
df['is_trip_length_sweet_spot'] = df['trip_duration_days'].apply(lambda x: 1 if 4 <= x <= 6 else 0)

# Calculate miles_per_day 
df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']

# Mileage efficiency optimal band 
df['mileage_efficiency_optimal'] = df['miles_per_day'].apply(lambda x: 1 if 180 <= x <= 220 else 0)

# Spending per day
df['spending_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']

# Spending categories based on trip length
def spending_category(row):
    days = row['trip_duration_days']
    spend = row['spending_per_day']
    if days <= 3:
        return 1 if spend <= 75 else 0
    elif 4 <= days <= 6:
        return 1 if spend <= 120 else 0
    else:
        return 1 if spend <= 90 else 0

df['spending_per_day_category'] = df.apply(spending_category, axis=1)

df['vacation_penalty_flag'] = np.where(
    (df['trip_duration_days'] >= 8) & (df['spending_per_day'] > 90), 1, 0)



feature_cols = [
    'trip_duration_days',
    'miles_traveled',
    'total_receipts_amount',
    'miles_per_day',
    'receipts_per_day',
    'receipts_per_mile',
    'is_trip_length_sweet_spot',
    'mileage_efficiency_optimal',
    'spending_per_day',
    'spending_per_day_category',
    'vacation_penalty_flag'
]

target_col = 'expected_output'

X = df[feature_cols]
y = df[target_col]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))


    for units in [256, 256, 128, 128, 64, 64]:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1))  

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

model = build_model(X_scaled.shape[1])


history = model.fit(
    X_scaled, y,
    epochs=400,
    batch_size=32,
    verbose=1
)


loss, mae = model.evaluate(X_scaled, y, verbose=0)
print(f"\nEvaluation on full training data:")
print(f"Mean Squared Error: {loss:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")


model.save('reimbursement_model.h5')
'''



#MODEL USE
import sys
# Load the saved model
model = tf.keras.models.load_model('reimbursement_model.h5')


trip_duration_days = float(sys.argv[1])
miles_traveled = float(sys.argv[2])
total_receipts_amount = float(sys.argv[3])


miles_per_day = miles_traveled / trip_duration_days
receipts_per_day = total_receipts_amount / trip_duration_days
receipts_per_mile = total_receipts_amount / miles_traveled if miles_traveled != 0 else 0

is_trip_length_sweet_spot = 1 if 4 <= trip_duration_days <= 6 else 0
mileage_efficiency_optimal = 1 if 180 <= miles_per_day <= 220 else 0
spending_per_day = total_receipts_amount / trip_duration_days

if trip_duration_days <= 3:
    spending_per_day_category = 1 if spending_per_day <= 75 else 0
elif 4 <= trip_duration_days <= 6:
    spending_per_day_category = 1 if spending_per_day <= 120 else 0
else:
    spending_per_day_category = 1 if spending_per_day <= 90 else 0

vacation_penalty_flag = 1 if (trip_duration_days >= 8 and spending_per_day > 90) else 0


features = np.array([[
    trip_duration_days,
    miles_traveled,
    total_receipts_amount,
    miles_per_day,
    receipts_per_day,
    receipts_per_mile,
    is_trip_length_sweet_spot,
    mileage_efficiency_optimal,
    spending_per_day,
    spending_per_day_category,
    vacation_penalty_flag
]])

# Predict
predicted = model.predict(features, verbose=0)
print(float(predicted[0][0]))

