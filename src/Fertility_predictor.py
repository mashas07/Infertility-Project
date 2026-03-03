import pandas as pd

from src.Fertility_model import FertilityModel

class FertilityPredictor(FertilityModel):
    # extends FertilityModel. Adds patient input validation and patient prediction
    ''' Extends FertilityModel class with input validation and patient prediction '''
    def __init__(self):
       '''  Initializes the FertilityPredictor and calls the parent FertilityModel initializer and sets up a container to store the last prediction result.
       Returns:
            None
       '''
       super().__init__()
       self._last_result: dict | None = None

    @property
    def last_result(self):
        ''' Gets the last result produced result by this predictor
        Returns:
        dict | None: The last result dictionary if a prediction has been made,
        otherwise None.
        '''
        return self._last_result
    # provides the summary of model performance

    def __str__(self):
        ''' Provide a readable result of the predictor
        If the model is not trained, returns a short message.
        If trained, includes accuracy, number of features, and the last prediction (if available).
        Returns:
        str: A summary string describing the predictor's status and performance.
        '''
        if not self._is_trained:
            return "Predictor is not trained yet"
        result = (
            f"PatientPredictor\n"
            f"  Accuracy : {self._accuracy * 100:.2f}%\n"
            f"  Features : {len(self._original_features)}")
        if self._last_result:
            result += f"\n  Last prediction : {self._last_result['prediction']} ({self._last_result['confidence']:.1f}%)"

        return result

    @staticmethod
    def _display_result(result: dict):
        ''' Print formatted prediction report to the console
        Args:
          result (dict): A result dictionary containing:
                - "prediction" (str): Predicted class label
                - "confidence" (float): Confidence percentage
                - "probabilities" (dict | None): Mapping of class -> probability percentage
        Returns:
            None
        '''
        print(f"\n{'*' * 70}")
        print("PREDICTION RESULTS")
        print(f"{'*' * 70}")
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']:.2f}%")
        if result['probabilities']:
            print("\n  Probability breakdown:")
            for cls, prob in result['probabilities'].items():
                bar = "=" * int(prob / 2)
                print(f"    {cls:20s} {bar} {prob:.1f}%")
        print(f"\n{'*' * 70}")
        print("!!  This is a prediction, not a medical diagnosis  !!")
        print("Please consult a qualified healthcare professional.")
        print(f"{'*' * 70}\n")

    def predict_patient(self, patient_data: dict) -> dict:
        ''' Makes a prediction using patient's data
        Args:
            patient_data (dict): Mapping of feature name -> value.
                Must contain exactly the features in self._original_features.
        Returns:
            dict: A result dictionary containing:
                - "prediction" (str): Decoded predicted class label
                - "confidence" (float): Confidence percentage
                - "probabilities" (dict): Mapping of class label -> probability percentage
                - "patient_data" (dict): The validated/converted patient input data
        '''
        # make a prediction using patient's data
        if not self._is_trained:
            raise RuntimeError("Predictor is not trained yet")
        #Check that all features are present
        if set(self._original_features) != set(patient_data.keys()):
            raise ValueError(f"Missing features: {set(self._original_features) - set(patient_data.keys())}")
        #Check that the data is numeric
        for feature in self._original_features:
            try:
                patient_data[feature] = float(patient_data[feature])
            except (ValueError, TypeError):
                raise ValueError(f"Feature '{feature}' must be numeric, got: {patient_data[feature]!r}")


        user_df = pd.DataFrame([patient_data])
        raw_pred, probs = self.predict_encoded(user_df)
        predicted_class, confidence, prob_dict = self.decode_prediction(raw_pred, probs)

        result = {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "patient_data": patient_data
        }

        self._last_result = result
        return result

    def interactive_prediction(self):
        ''' Run an interactive command-line session to collect patient's data
        Returns:
            None
        '''
        print("Patient Data")
        print("*" * 70)
        print("Enter 1 (yes) or 0 (no) for each question, except for age")
        patient_data = {}
        
        for feature in self._original_features:
            value = input(f"{feature}: ")
            if feature == 'Age':
                try:
                    patient_data[feature] = float(value)
                    break
                except ValueError:
                    print("Please enter a valid number for age")
            else:
                if value in ('0', '1'):
                    patient_data[feature] = float(value)
                    break
                else:
                    print("Please enter 0 or 1")

        print(f"\nProcessing patient data")
        result = self.predict_patient(patient_data)
        self._last_result = result
        self._display_result(result)



