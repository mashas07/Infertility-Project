import pandas as pd

from src.Fertility_model import FertilityModel

class FertilityPredictor(FertilityModel):
    '''Extends FertilityModel via inheritance.
    Adds patient input validation and patient prediction.'''

    def __init__(self):
        super().__init__()
        self._last_result: dict | None = None

    @property
    def last_result(self):
        return self._last_result

    def __str__(self):
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
        print("     Please consult a qualified healthcare professional.")
        print(f"{'*' * 70}\n")

    def predict_patient(self, patient_data: dict) -> dict:
        '''Make a prediction given patient data'''
        #Check that the model is trained
        if not self._is_trained:
            raise  RuntimeError("Predictor is not trained yet")
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
        print("Patient Data")
        print(f"{"*" * 70}")

        patient_data = {}
        for feature in self._original_features:
    while True:
        raw = input(f"{feature}: ").strip()
        if not raw:
            print(f"! '{feature}' is required.")
            continue
        try:
            value = float(raw)
            if feature.lower() != 'age' and value not in (0, 1):
                print(f"! '{feature}' must be 0 or 1.")
                continue
            patient_data[feature] = value
            break
        except ValueError:
            print(f"! Please enter a valid number.")

        print(f"\nProcessing patient data")
        result = self.predict_patient(patient_data)
        self._last_result = result
        self._display_result(result)



