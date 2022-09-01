from keras.models import model_from_json
import numpy as np

class EmotionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # Model is loaded in using metadata from JSON
        with open(model_json_file, "r") as json:
            loaded_model_as_json = json.read()
            self.loaded_model = model_from_json(loaded_model_as_json) # Model loaded in from JSON

        self.loaded_model.load_weights(model_weights_file) # The parameters set out in weights loaded into model

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img) # The model with the model weight configuration are used on our frames
        return EmotionModel.EMOTIONS_LIST[np.argmax(self.preds)] # Numpy extracts maximum indices for emotion
