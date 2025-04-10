import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("ethiopian_students_dataset_final.csv")

# Remove columns that should not be asked
if "Satisfaction_Score" in df.columns:
    df.drop(columns=["Satisfaction_Score"], inplace=True)
if "Full_Name" in df.columns:
    df.drop(columns=["Full_Name"], inplace=True)

# Automatically detect categorical columns and encode them
label_encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define target variable
target = "Department"
target_encoder = LabelEncoder()
df[target] = target_encoder.fit_transform(df[target])

# Features to be used for modeling
selected_features = [
    "Age", "Total_Academic_Assessment", "Mathematics_Proficiency", "Physics_Proficiency", 
    "Programming_Experience", "Problem_Solving_Ability", "Technical_Skill_Interest", "Creativity_Level",
    "Preferred_Work_Type", "High_School_Background", "Communication_Skills", "Preferred_Future_Career",
    "Location_Preference", "Drawing_Design_Skills", "Climate_Science_Interest", "Field_Survey_Interest",
    "Construction_Infrastructure_Interest"
]

# Split dataset
X = df[selected_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Prompt user for input values
print("\nPlease answer the following questions:")
user_input = {}
questions = {
    "Age": "What is your age (number between 20-26)? ",
    "Total_Academic_Assessment": "What is your Total Academic Assessment (%)? ",
    "Mathematics_Proficiency": "What is your Mathematics Proficiency (1-100)? ",
    "Physics_Proficiency": "What is your Physics Proficiency (1-100)? ",
    "Programming_Experience": "What is your Programming Experience (No, Beginner, Intermediate, Advanced)? ",
    "Problem_Solving_Ability": "Rate your Problem-Solving Ability (1-5)? ",
    "Technical_Skill_Interest": "How interested are you in Technical Skills? (1-5)? ",
    "Creativity_Level": "Rate your Creativity Level (1-5)? ",
    "Preferred_Work_Type": "Do you prefer Field Work, Office Work, or Hybrid? ",
    "High_School_Background": "What was your high school background? (Science, Social, Technical)? ",
    "Communication_Skills": "Rate your Communication Skills (1-5)? ",
    "Preferred_Future_Career": "What is your preferred future career? ",
    "Location_Preference": "Do you prefer to work in Urban, Rural, or No Preference? ",
    "Drawing_Design_Skills": "Rate your Drawing/Design Skills (1-5)? ",
    "Climate_Science_Interest": "Are you interested in Climate Science? (Yes/No) ",
    "Field_Survey_Interest": "Do you like field surveys? (Yes/No) ",
    "Construction_Infrastructure_Interest": "Are you interested in Construction and Infrastructure? (Yes/No) "
}

for feature in selected_features:
    question = questions.get(feature, f"What is your {feature.replace('_', ' ')}? ")
    if feature in categorical_columns:
        valid_values = [v.lower() for v in label_encoders[feature].classes_.tolist()]
        user_value = input(f"{question} Choose from {valid_values}: ").strip().lower()
        while user_value not in valid_values:
            print("Invalid input. Please enter one of the valid options.")
            user_value = input(f"{question} Choose from {valid_values}: ").strip().lower()
        user_input[feature] = label_encoders[feature].transform([user_value.capitalize()])[0]
    else:
        user_input[feature] = float(input(question).strip())

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Make prediction
predicted_department_code = model.predict(input_df)[0]
predicted_department = target_encoder.inverse_transform([predicted_department_code])[0]

# Explain recommendation using feature importances
feature_importances = dict(zip(selected_features, model.feature_importances_))
important_factors = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:3]
explanation = f"The recommendation is {predicted_department} because: "
explanation += "\n".join([f"- {factor[0].replace('_', ' ')}: Your score influenced the decision significantly." for factor in important_factors])

# Display results
print(f"\nRecommended Department: {predicted_department}")
print(f"Reasons:\n{explanation}")
