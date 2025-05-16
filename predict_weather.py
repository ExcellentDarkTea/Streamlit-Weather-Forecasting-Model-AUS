import joblib
import pandas as pd

class WeatherModel:
    def __init__(self, model_path):
        self.model_bundle = joblib.load(model_path)
        self.model = self.model_bundle['model']
        self.imputer = self.model_bundle['imputer']
        self.scaler = self.model_bundle['scaler']
        self.encoder = self.model_bundle['encoder']
        self.input_cols = self.model_bundle['input_cols']
        self.target_col = self.model_bundle['target_col']
        self.numeric_cols = self.model_bundle['numeric_cols']
        self.categorical_cols = self.model_bundle['categorical_cols']
        self.encoded_cols = self.model_bundle['encoded_cols']

    def preprocess(self, input_data: dict) -> pd.DataFrame:
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=self.input_cols)

        # Impute
        input_df[self.numeric_cols] = self.imputer.transform(input_df[self.numeric_cols])

        # Scale
        input_df[self.numeric_cols] = self.scaler.transform(input_df[self.numeric_cols])

        # Encode
        encoded_cats = self.encoder.transform(input_df[self.categorical_cols])
        encoded_cat_df = pd.DataFrame(encoded_cats, columns=self.encoded_cols)

        # Combine
        X = pd.concat([
            input_df[self.numeric_cols].reset_index(drop=True),
            encoded_cat_df.reset_index(drop=True)
        ], axis=1)
        return X

    def predict(self, input_data: dict):
        X = self.preprocess(input_data)
        prediction = self.model.predict(X)
        prediction_proba = self.model.predict_proba(X)
        return prediction[0], prediction_proba[0][1]