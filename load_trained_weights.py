import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Definir la clase del modelo
class NClassifier(nn.Module):
    def __init__(self, architecture):
        super(NClassifier, self).__init__()
        self.encoder = nn.Sequential(*architecture)

    def forward(self, x):
        return self.encoder(x)

# Cargar datos y preparar encoder
dataset = pd.read_csv('/Users/samanthacovarrubiasfigueroa/Documents/progra/Py/crops/Crop_recommendation.csv')
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

print("Intentando cargar modelo entrenado...")

# Intentar cargar el modelo original entrenado
try:
    # Cargar usando diferentes métodos
    original_model = torch.load('modelo_crop.pkl', map_location='cpu', weights_only=False)
    
    if hasattr(original_model, 'state_dict'):
        # Si es el modelo directamente
        model.load_state_dict(original_model.state_dict())
        print("Cargado desde modelo directo")
    elif isinstance(original_model, dict):
        if 'model_state_dict' in original_model:
            # Si es un diccionario con state_dict
            model.load_state_dict(original_model['model_state_dict'])
            print("Cargado desde diccionario con state_dict")
        elif 'model_architecture' in original_model:
            # Si tiene la arquitectura completa
            if hasattr(original_model['model_architecture'], 'state_dict'):
                model.load_state_dict(original_model['model_architecture'].state_dict())
                print("Cargado desde modelo en diccionario")
            else:
                print("No se encontraron pesos entrenados, usando modelo sin entrenar")
        else:
            print("Formato no reconocido, usando modelo sin entrenar")
    else:
        print("Formato no compatible, usando modelo sin entrenar")
        
except Exception as e:
    print(f"Error cargando modelo original: {e}")
    print("Usando modelo sin entrenar")

# Poner el modelo en modo evaluación
model.eval()

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

print("Modelo final guardado en 'modelo_crop_api.pkl'")

# Probar el modelo con un ejemplo
test_features = [90, 42, 43, 20.9, 82.0, 6.5, 202.9]  # Ejemplo de rice
input_tensor = torch.tensor(test_features, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = output.argmax(dim=1).item()
    confidence = probabilities.max().item()

predicted_crop = crop_code.inverse_transform([predicted_class])[0]

print(f"\nPrueba del modelo:")
print(f"   Entrada: {test_features}")
print(f"   Predicción: {predicted_crop}")
print(f"   Confianza: {confidence:.3f}")
print(f"   Clase predicha: {predicted_class}")

print(f"\nCultivos disponibles: {len(crop_code.classes_)}")
print("Listo para usar con la API!")