import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Definir la clase del modelo (copia exacta de tu script original)
class NClassifier(nn.Module):
    def __init__(self, architecture):
        super(NClassifier, self).__init__()
        self.encoder = nn.Sequential(*architecture)

    def forward(self, x):
        return self.encoder(x)

# Cargar datos y preparar encoder (igual que en tu script original)
dataset = pd.read_csv('Crop_recommendation.csv')
label = dataset["label"]
crop_code = LabelEncoder()
label_encoded = crop_code.fit_transform(label)

# Definir la mejor arquitectura (la #2 que tuvo 95.45% accuracy)
best_architecture = [
    nn.Linear(7, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 22)
]

# Crear el modelo
model = NClassifier(best_architecture)

# Cargar el estado del modelo guardado anteriormente
try:
    # Intentar cargar el modelo anterior
    checkpoint = torch.load('modelo_crop.pkl', map_location='cpu')
    if hasattr(checkpoint, 'state_dict'):
        model.load_state_dict(checkpoint.state_dict())
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Formato de modelo no reconocido, usando modelo sin entrenar")
except:
    print("No se pudo cargar modelo anterior, usando modelo sin entrenar")

# Guardar correctamente para la API
model_data = {
    'model_state_dict': model.state_dict(),
    'crop_encoder': crop_code,
    'architecture': best_architecture,
    'input_size': 7,
    'output_size': 22
}

# Guardar con pickle
with open('modelo_crop_api.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Modelo guardado correctamente para la API en 'modelo_crop_api.pkl'")
print("Cultivos disponibles:")
for i, crop in enumerate(crop_code.classes_):
    print(f"  {i}: {crop}")