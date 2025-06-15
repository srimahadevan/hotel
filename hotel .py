import pandas as pd      
import numpy as np          
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import re  # For input validation

# ✅ Load Dataset
file_path = r"C:\Users\nithi\OneDrive\Desktop\ML PROJECT\REGRESSION MODELS\hotel\hotel_bookings.csv"
df = pd.read_csv(file_path)
print("✅ File Loaded Successfully!\n")

# ✅ Loop until valid input for dataset preview
while True:
    to_see_the_data = input("ENTER 'yes' or 'no' TO SEE THE DATA: ").strip().lower()
    if to_see_the_data in ["yes", "y"]:
        print("\nDataset Preview:\n", df.head())
        break
    elif to_see_the_data in ["no", "n"]:
        print("✅ Skipping dataset preview. Moving to the next step...\n")
        break
    else:
        print("⚠️ Invalid input! Please enter 'yes' or 'no'.\n")

# ✅ Loop until valid input for dataset info
while True:
    to_see_the_info = input("ENTER 'yes' or 'no' TO SEE THE DATASET INFO: ").strip().lower()
    if to_see_the_info in ["yes", "y"]:
        print("\nDataset Info:\n")
        df.info()
        break
    elif to_see_the_info in ["no", "n"]:
        print("✅ Skipping dataset info. Moving to the next step...\n")
        break
    else:
        print("⚠️ Invalid input! Please enter 'yes' or 'no'.\n")

# ✅ Label Encoding for Categorical Features
categorical_features = [
    "hotel", "arrival_date_month", "meal", "country", 
    "market_segment", "distribution_channel", "reserved_room_type", 
    "assigned_room_type", "deposit_type", "customer_type", 
    "reservation_status", "reservation_status_date"
]

label_encoders = {}  
for col in categorical_features:
    if col in df.columns:  # ✅ Ensure column exists before encoding
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le  # ✅ Store encoder
    else:
        print(f"⚠️ Warning: Column '{col}' not found in the dataset!")

# ✅ Define target variables
target_features = ["adr", "lead_time", "stays_in_week_nights"]
y = df[target_features]

# ✅ Define features (X) by dropping target variables
X = df.drop(columns=target_features)

# ✅ Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data Split Completed!\n")

# ✅ Train the XGBoost MultiOutput Regressor
model = MultiOutputRegressor(xgb.XGBRegressor(objective="reg:squarederror", 
                                              n_estimators=100, 
                                              learning_rate=0.1))
model.fit(X_train, y_train)

print("✅ Model Training Completed!\n")

# ✅ Define feature categories
numerical_features = [
    "lead_time", "arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month", 
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", 
    "previous_cancellations", "previous_bookings_not_canceled", "booking_changes"
]

categorical_features = [
    "hotel", "meal", "country", "market_segment", "distribution_channel", "is_repeated_guest", 
    "reserved_room_type", "assigned_room_type"
]

# ✅ Combine all features
feature_names = numerical_features + categorical_features

# ✅ Dictionary to store user inputs
X_input = {}

# ✅ Collect user input
for feature in feature_names:
    while True:
        value = input(f"Enter value for {feature} (or 'no' if unknown): ").strip()
        
        # ✅ Validate numerical features (Only allow numbers)
        if feature in numerical_features:
            if value.lower() == "no":
                print(f"⚠️ You must provide a value for {feature}!")
                continue  # Ask again
            elif re.match(r"^-?\d+(\.\d+)?$", value):  
                X_input[feature] = float(value)
                break
            else:
                print(f"⚠️ Invalid input for {feature}! Please enter a numeric value.")

        # ✅ Validate categorical features (Only allow valid labels)
        elif feature in categorical_features:
            le = label_encoders.get(feature)

            if le is None:
                print(f"⚠️ No encoder found for '{feature}'. Skipping...")
                continue

            if value.lower() == "no":
                print(f"⚠️ You must provide a value for {feature}!")
                continue  # Ask again

            if re.match(r"^[a-zA-Z\s]+$", value):  # ✅ Allows letters & spaces
                if value in le.classes_:  # ✅ Check if value exists in LabelEncoder
                    X_input[feature] = value
                    break
                else:
                    print(f"⚠️ The input '{value}' does not match any known category for '{feature}'. Please check your input and enter one of: {list(le.classes_)}")
            else:
                print(f"⚠️ The input '{value}' is not a valid category for '{feature}'. Please enter a proper name using only letters and spaces.")

print("\n✅ Input Features Collected Successfully!")
print(X_input)

# ✅ Add target columns explicitly to X_input with NaN
for target in target_features:
    X_input[target] = np.nan

# ✅ Convert input into a DataFrame
X_input_df = pd.DataFrame([X_input])

# ✅ Encode categorical values using the stored LabelEncoders
for col, le in label_encoders.items():
    if col in X_input_df.columns and le is not None:
        try:
            X_input_df[col] = le.transform(X_input_df[col].astype(str))
        except ValueError:
            print(f"⚠️ Error: The input '{X_input_df[col].values[0]}' for '{col}' is not in the training data. Please enter a valid category from {list(le.classes_)}.")
            exit()  # Stop execution if an invalid category is entered

# ✅ Handle missing target values by predicting them
unknown_targets = [target for target in target_features if pd.isna(X_input_df[target].values[0])]

if unknown_targets:
    X_input_df[unknown_targets] = model.predict(X_input_df.drop(columns=target_features, errors="ignore"))
    print("\n✅ Predicted Missing Values:")
    for target in unknown_targets:
        print(f"{target}: {X_input_df[target].values[0]:.2f}")

# ✅ Make predictions using trained model
y_pred = model.predict(X_test)

# ✅ Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
print(f"\n✅ Model MAE (lower is better): {mae:.2f}")
