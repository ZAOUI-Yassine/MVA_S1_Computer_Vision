from data import train_transforms, val_transforms
from model import Net

class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.train_transform = self.init_train_transform()
        self.val_transform = self.init_val_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        else:
            raise NotImplementedError("Model not implemented")

    def init_train_transform(self):
        if self.model_name == "basic_cnn":
            return train_transforms
        else:
            raise NotImplementedError("Train Transform not implemented")

    def init_val_transform(self):
        if self.model_name == "basic_cnn":
            return val_transforms
        else:
            raise NotImplementedError("Validation Transform not implemented")

    def get_model(self):
        return self.model

    def get_transforms(self):
        return self.train_transform, self.val_transform

    def get_all(self):
        return self.model, self.train_transform, self.val_transform
    def get_alle(self):
        return self.model, self.val_transform