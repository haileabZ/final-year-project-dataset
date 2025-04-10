# predict.py
import joblib
import pandas as pd

def predict_department():
    # Load saved artifacts
    artifacts = joblib.load('department-model.joblib')
    
    # Extract components
    model = artifacts['best_model']
    label_encoders = artifacts['label_encoders']
    target_encoder = artifacts['target_encoder']
    categorical_columns = artifacts['categorical_columns']
    selected_features = artifacts['selected_features']
    
    # Collect user input
    user_input = {}
    questions = {
        "Age": "What is your age (number between 20-26)? ",
        # ... (include all your questions here)
    }
    
    for feature in selected_features:
        question = questions.get(feature, f"What is your {feature.replace('_', ' ')}? ")
        if feature in categorical_columns:
            valid_values = [v.lower() for v in label_encoders[feature].classes_.tolist()]
            while True:
                user_value = input(f"{question} Choose from {valid_values}: ").strip().lower()
                if user_value in valid_values:
                    user_input[feature] = label_encoders[feature].transform([user_value.capitalize()])[0]
                    break
                print("Invalid input. Please try again.")
        else:
            user_input[feature] = float(input(question).strip())
    
    # Create DataFrame and predict
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)
    department = target_encoder.inverse_transform(prediction)[0]
    
    print(f"\nRecommended Department: {department}")

if __name__ == "__main__":
    predict_department()