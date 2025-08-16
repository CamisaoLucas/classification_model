import keras
from autokeras.preprocessors.common import AddOneDimension, CastToString
from autokeras.preprocessors.encoders import OneHotEncoder

# Registrar custom layers/objects usados pelo AutoKeras
custom_objects = {
    "AddOneDimension": AddOneDimension,
    "CastToString": CastToString,
    "OneHotEncoder": OneHotEncoder,
}

# Carregar modelo com suporte a objetos customizados
modelo = keras.models.load_model("best_model.h5", custom_objects=custom_objects)

# Conferir arquitetura
modelo.summary()
