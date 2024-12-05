import streamlit as st
import pandas as pd
import pickle  # For loading the saved model
from sklearn.preprocessing import StandardScaler

def predict_with_knn(features, scaler, model, feature_names):
    # Convert features to a DataFrame with appropriate feature names
    features_df = pd.DataFrame([features], columns=feature_names)
    
    # Standardize the input data using the provided scaler
    input_data_scaled = scaler.transform(features_df)

    # Predict the class using the scaled data
    predicted_class = model.predict(input_data_scaled)
    return predicted_class[0]  # Convert to a hashable type

def main():
    st.title("KNN Prediction App")

    # Mapping of class values to Target descriptions
    target_mapping = {
        1: "1phase3%fault50%load", 2: "2phase3%fault50%load", 3: "3phase3%fault50%load", 
        4: "2phase1%fault50%load", 5: "2phase1%fault75%load", 6: "2phase1%fault100%load", 
        7: "2phase1%fault0%load", 8: "2phase3%fault100%load", 9: "0%fault50%load", 
        10: "0%fault75%load", 11: "0%fault100%load", 12: "0%fault0%load", 
        13: "1phase1%fault50%load", 14: "1phase1%fault75%load", 15: "1phase1%fault100%load", 
        16: "1phase1%fault0%load", 17: "1phase3%fault75%load", 18: "1phase3%fault100%load", 
        19: "3phase3%fault75%load", 20: "3phase3%fault0%load", 21: "3phase1%fault75%load", 
        22: "3phase1%fault0%load", 23: "3phase3%fault100%load", 24: "2phase3%fault75%load", 
        25: "2phase3%fault0%load", 26: "1phase3%fault0%load", 27: "3phase1%fault100%load"
    }

    # Load the saved model using pickle
    model_path = "Knn_Model_v2.pkcls"  # Replace with the actual path
    with open(model_path, 'rb') as file:
        knn_model = pickle.load(file)

    # Create a new StandardScaler instance
    scaler = StandardScaler()

    # Define the required columns
    required_columns = [
        'Rotor Flux_B', 'Speed', 'RUL', 'Ias_RMS', 'Ibs_RMS', 
        'TPhase A', 'TPhase B', 'Rotor Flux_A', 
        'Ics_RMS', 'TPhase C', 'Rotor Flux_C'
    ]

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Check for required columns in the uploaded file
        if not all(col in df.columns for col in required_columns):
            st.error("The uploaded CSV file does not contain all the required columns.")
            return

        # Fit the scaler to the entire dataset to ensure consistency
        scaler.fit(df[required_columns])

        # Initialize a dictionary to store predicted class frequencies
        prediction_counts = {}

        # Iterate over each row and predict
        for index, row in df.iterrows():
            features = [
                row['Rotor Flux_B'], row['Speed'], row['RUL'], 
                row['Ias_RMS'], row['Ibs_RMS'], row['TPhase A'], 
                row['TPhase B'], row['Rotor Flux_A'], row['Ics_RMS'], 
                row['TPhase C'], row['Rotor Flux_C']
            ]
            
            predicted_class = predict_with_knn(features, scaler, knn_model, required_columns)

            # Convert predicted_class to int (or str if needed)
            predicted_class = int(predicted_class)  # Ensure it's a hashable type

            # Increment the count for the predicted class
            prediction_counts[predicted_class] = prediction_counts.get(predicted_class, 0) + 1

        # Find the most frequent predicted class
        max_class = max(prediction_counts, key=lambda k: prediction_counts[k])
        max_frequency = prediction_counts[max_class]

        # Map the class to its Target description
        target_description = target_mapping.get(max_class, "Unknown Class")

        # Display the result
        st.write(f"The most frequent predicted class is '{max_class}' with a frequency of {max_frequency}.")
        st.write(f"This corresponds to the target: **{target_description}**.")

        # Clear the prediction dictionary
        prediction_counts.clear()

if __name__ == "__main__":
    main()
