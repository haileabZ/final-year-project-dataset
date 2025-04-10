# train.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load and preprocess data
def load_and_preprocess():
    df = pd.read_csv("ethiopian_students_dataset_final.csv")
    
    # Remove unnecessary columns
    for col in ["Satisfaction_Score", "Full_Name"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    # Encode categorical features
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders, categorical_columns

# Main training function
def train_and_save_model():
    df, label_encoders, categorical_columns = load_and_preprocess()
    
    # Define target variable
    target = "Department"
    target_encoder = LabelEncoder()
    df[target] = target_encoder.fit_transform(df[target])
    
    # Feature selection
    selected_features = [
        "Age", "Total_Academic_Assessment", "Mathematics_Proficiency", 
        "Physics_Proficiency", "Programming_Experience", "Problem_Solving_Ability",
        "Technical_Skill_Interest", "Creativity_Level", "Preferred_Work_Type",
        "High_School_Background", "Communication_Skills", "Preferred_Future_Career",
        "Location_Preference", "Drawing_Design_Skills", "Climate_Science_Interest",
        "Field_Survey_Interest", "Construction_Infrastructure_Interest"
    ]
    
    # Split data
    X = df[selected_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = [
        ('Random Forest', RandomForestClassifier(random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Logistic Regression', LogisticRegression(max_iter=1000)),
        ('SVM', SVC())
    ]
    
    # Train and evaluate models
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    
    print(f"\nBest Model: {best_model_name} with accuracy {best_accuracy:.4f}")
    
    # Save artifacts
    artifacts = {
        'best_model': best_model,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'categorical_columns': categorical_columns,
        'selected_features': selected_features
    }
    
    joblib.dump(artifacts, 'model_artifacts.joblib')
    print("\nModel and encoders saved successfully!")

if __name__ == "__main__":
    train_and_save_model()