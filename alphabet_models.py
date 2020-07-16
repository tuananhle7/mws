import models


class GenerativeModel(models.GenerativeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, use_alphabet=True)


class InferenceNetwork(models.InferenceNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, use_alphabet=True)
