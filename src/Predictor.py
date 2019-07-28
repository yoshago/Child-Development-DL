import numpy as np
import pickle
from keras.models import load_model

class Predictor:
    def __init__(self,model_file_name, test_names):
        self.model_file_name=model_file_name
        self.model = self.load_model()
        self.test_names=np.array(test_names)
    def load_model(self):
        return load_model(self.model_file_name)


    def test(self):
        for j in range(1, 95):
            with open('Data/compressed_data/data' + str(j) + '.txt', 'rb') as video:
                test_records = np.array(pickle.load(video))
                if (int(test_records[0].name) in self.test_names):
                    for i in test_records:
                        tmp = np.array(i.matrix).reshape(50 * 85 * 85)
                        x_test.append(tmp)
                    x_test = np.array(x_test)
                    predictions = self.model.predict(x_test)
                    print("suppose to be:" + str(test_records[0].label))
                    print(predictions)
                    x_test = []
