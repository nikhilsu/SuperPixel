from gan_model.config import Config


class GANComponent(object):
    def __init__(self, model):
        self.model = model

    def _compile(self, loss='mean_absolute_error', optimizer=Config.adam_optimizer, metrics=None):
        self.model.compile(loss=loss, optimizer=optimizer(), metrics=metrics)

    def setup_input_tensor(self, tensor):
        return self.model(tensor)

    def train_on_batch(self, input_tensor, output_tensor):
        return self.model.train_on_batch(input_tensor, output_tensor)

    def predict(self, input_tensor):
        return self.model.predict(input_tensor)
