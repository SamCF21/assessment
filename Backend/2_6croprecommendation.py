import os
import torch
import numpy as np
import pandas as pd
from torch import optim, nn, utils, Tensor
from torch.utils.data import random_split, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

dataset = pd.read_csv('Crop_recommendation.csv')
characteristics = dataset[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
label = dataset["label"]
# print(label)

# El modelo no puede operar con texto, por lo cual desginamos un valor numérico a cada label.

crop_code = LabelEncoder() # Crea el codificador de etiquetas
label = crop_code.fit_transform(label) # Se reemplaza cada label por un número
# print(label)

x_tensor = torch.tensor(characteristics.values, dtype=torch.float32) # Se convierten las caracteristicas a tensor a floats
y_tensor = torch.tensor(label, dtype=torch.long) # Se convierten las labels a enteros tipo long
full_dataset = TensorDataset(x_tensor, y_tensor) # Se unen las caracteristicas y las labels
# print(y_tensor[:5])

total_size = len(full_dataset) # Dataset completpo
train_size = int(0.7 * total_size) #70% para entrenar
val_size = int(0.15 * total_size) # 15% para validar
test_size = int(0.15 * total_size) # 15% para probar
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size]) # Se divide aleatoriamente el dataset en los tamaños definidos

# Se crean los DataLoaders para los tres subconjuntos
# División en batches de 64 para mayor eficiencia
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Se utiliza shuffle para mezclar los datos en cada epoch
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

architectures = [
    # 1 capa oculta
    [nn.Linear(7, 128), # Capa lineal
     nn.ReLU(), # Función de activación, rompe linealidad y es más veloz
     nn.Linear(128, 22)], # salida de los 22 posibles cultivos

    # 2 capas ocultas
    [nn.Linear(7, 128),
     nn.ReLU(),
     nn.Linear(128, 64),
     nn.ReLU(),
     nn.Linear(64, 22)],

    # 3 capas ocultas
    [nn.Linear(7, 256),
     nn.ReLU(),
     nn.Linear(256, 128),
     nn.ReLU(),
     nn.Linear(128, 64),
     nn.ReLU(),
     nn.Linear(64, 22)],

]

# Definición de clase para el modelo de clasificación
class NClassifier(nn.Module): # Constructor
  def __init__(self, architecture):
    super(NClassifier, self).__init__()
    self.encoder = nn.Sequential(*architecture) # Se crea la red neuronal, utilizamos *arquitecture para probar las diferentes arquitecturas y leer adecuadmaente cada una de sus capas

  def forward(self, x): # Se establece el paso hacia adelante
    return self.encoder(x) # Se pasan los datos por cada capa

# Función para impresión de matriz de confusión
def print_confusion_matrix(tensor, class_labels=None):
    matrix = tensor.cpu().numpy() # Se pasa a un array de numpy y al CPU en dado caso de que haya estado usando GPU
    n = matrix.shape[0] # Cantidad de clases
    if class_labels is None: # Se verifica que hayan clases
        class_labels = [str(i) for i in range(n)] # En dado caso que no, se emplean índices
    bold = lambda text: f"\033[1m{text}\033[0m" # Impresión en negritas
    max_width = max(len(label) for label in class_labels + [str(int(matrix.max()))]) + 2 # Se determina el ancho necesario para que se impriman adecuadamente todas las columnas
    print("\n" + " " * max_width + bold("Matriz de confusión (filas = esperados, columnas = predichos)")) # Se imprime el título
    aligned_labels = [f"{label:>{max_width}}" for label in class_labels] # Se alinean las etiquetas de columnas
    header = " " * max_width + "".join(bold(label) for label in aligned_labels)
    print(header)
    # bucle de impresión
    for i in range(n):
        row_label = f"{class_labels[i]:>{max_width}}" # Nombre de clases
        row_label = bold(row_label)  # Se ponen en negritas para mejor distinción
        row_data = "".join(f"{int(val):>{max_width}}" for val in matrix[i])
        print(row_label + row_data) # Se imprime

# bucle para iterar las distintas arquitecturas
for i in range(len(architectures)):
  torch.manual_seed(36) # Se establece la semilla a utilizar
  print(f"-----------------ARQUITECTURE {i+1} -----------------") # Título para distinguir entre arquitecturas
  model = NClassifier(architectures[i]) # Se crea el modelo para la arquitectura
  if(torch.cuda.is_available()): # Se verifica si el GPU está disponible para hacer el entrenamiento más rápido
    model.cuda(0)

  loss = nn.CrossEntropyLoss() # Se define la función de pérdida a utilizar
  params = model.parameters() # Se extraen los pesos y bias
  optimizer = optim.Adam(params, lr=1e-3)
  model.train()
  num_epochs = 40
  best_val_loss = float('inf')
  patience = 5
  wait = 0
  train_loss_history = []
  val_loss_history = []

  print("training on a tensor shaped", len(train_loader), "for epochs", num_epochs)


  for epoch in range(num_epochs):
      model.train()
      train_losses = []

      for x, y in train_loader:
          if torch.cuda.is_available():
              x = x.cuda(0)
              y = y.cuda(0)
          y_hat = model(x)
          loss = nn.functional.cross_entropy(y_hat, y)
          train_losses.append(loss.item())
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

      avg_train_loss = sum(train_losses) / len(train_losses)
      train_loss_history.append(avg_train_loss)

      # Validación
      model.eval()
      val_losses = []
      with torch.no_grad():
          for x, y in val_loader:
              if torch.cuda.is_available():
                  x = x.cuda(0)
                  y = y.cuda(0)
              y_hat = model(x)
              val_loss = nn.functional.cross_entropy(y_hat, y)
              val_losses.append(val_loss.item())

      avg_val_loss = sum(val_losses) / len(val_losses)
      val_loss_history.append(avg_val_loss)

      print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

      # Early stopping
      if avg_val_loss < best_val_loss:
          best_val_loss = avg_val_loss
          wait = 0
      else:
          wait += 1
          if wait >= patience:
              print(f"Early stopping en la época {epoch+1}")
              break

  results = []

  n_classes = 22  # Total de cultivos
  confusion_mtx = torch.zeros(n_classes, n_classes, dtype=torch.int32)

  model.eval()
  with torch.no_grad():
      for x, y in test_loader:
          if torch.cuda.is_available():
              x = x.cuda(0)
              y = y.cuda(0)
          y_hat = model(x)
          y_hat_softmax = nn.functional.softmax(y_hat, dim=1)
          preds = y_hat_softmax.argmax(dim=1)
          for t, p in zip(y, preds):
              confusion_mtx[t.long(), p.long()] += 1
  gen_precision = 0
  gen_recall = 0
  gen_f1 = 0

  total = confusion_mtx.sum().item()
  for i in range(n_classes):
      TP = confusion_mtx[i, i].item()
      FN = confusion_mtx[i, :].sum().item() - TP
      FP = confusion_mtx[:, i].sum().item() - TP
      TN = total - TP - FP - FN

      precision = TP / (TP + FP) if (TP + FP) > 0 else 0
      recall = TP / (TP + FN) if (TP + FN) > 0 else 0
      accuracy = (TP + TN) / total
      f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

      gen_precision += precision
      gen_recall += recall
      gen_f1 += f1

      results.append({
          "Clase": i,
          "TP": TP,
          "FP": FP,
          "FN": FN,
          "TN": TN,
          "Precision": round(precision, 4),
          "Recall": round(recall, 4),
          "F1-Score": round(f1, 4),
          "Accuracy": round(accuracy, 4)
      })

  print_confusion_matrix(confusion_mtx)

  df_metrics = pd.DataFrame(results)
  print("\nMétricas por clase:")
  print(df_metrics)

  # Promedios
  gen_precision /= n_classes
  gen_recall /= n_classes
  gen_f1 /= n_classes

  print("\nPromedios para todas las clases:")
  print(f"Precision: {gen_precision:.4f}")
  print(f"Recall: {gen_recall:.4f}")
  print(f"F1-Score: {gen_f1:.4f}")
  accuracy_total = (confusion_mtx.diag().sum().item() / total) * 100
  print(f"Accuracy: {accuracy_total:.2f}%")

#   plt.figure(figsize=(10, 6))
#   plt.plot(train_loss_history, label='Train Loss', marker='o')
#   plt.plot(val_loss_history, label='Validation Loss', marker='o')
#   plt.title("Pérdida de Entrenamiento vs Validación")
#   plt.xlabel("Época")
#   plt.ylabel("Loss")
#   plt.legend()
#   plt.grid(True)
#   plt.tight_layout()
#   plt.show()

# Funcion para proabr el modelo con ejemplos
def predict_crop(model, crop_encoder, input_features, use_cuda=True):
    model.eval()
    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)

    if torch.cuda.is_available() and use_cuda:
        input_tensor = input_tensor.cuda()
        model = model.cuda()

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    return crop_encoder.inverse_transform([predicted_class])[0]

#Ejemplos de prueba del modelo
sample_apple = [20.8, 134.22, 199.89, 22.63, 92.33, 5.93, 112.65]
sample_banana = [100.23, 82.01, 50.05, 27.38, 80.36, 5.98, 104.63]
sample_blackgram = [40.02, 67.47, 19.24, 29.97, 65.12, 7.13, 67.88]
sample_chickpea = [40.09, 67.79, 79.92, 18.87, 16.86, 7.34, 80.06]
sample_coconut = [21.98, 16.93, 30.59, 27.41, 94.84, 5.98, 175.69]
sample_coffee = [101.2, 28.74, 29.94, 25.54, 58.87, 6.79, 158.07]
sample_cotton = [117.77, 46.24, 19.56, 23.99, 79.84, 6.91, 80.4]
sample_grapes = [23.18, 132.53, 200.11, 23.85, 81.88, 6.03, 69.61]
sample_jute = [89.0, 47.0, 38.0, 25.52, 72.25, 6.0, 151.89]
sample_kidneybeans = [20.75, 67.54, 20.05, 20.12, 21.61, 5.75, 105.92]
sample_lentil = [18.77, 68.36, 19.41, 24.51, 64.8, 6.93, 45.68]
sample_maize = [77.76, 48.44, 19.79, 22.39, 65.09, 6.25, 84.77]
sample_mango = [20.07, 27.18, 29.92, 31.21, 50.16, 5.77, 94.7]
sample_mothbeans = [21.44, 48.01, 20.23, 28.19, 53.16, 6.83, 51.2]
sample_mungbean = [20.99, 47.28, 19.87, 28.53, 85.5, 6.72, 48.4]
sample_muskmelon = [100.32, 17.72, 50.08, 28.66, 92.34, 6.36, 24.69]
sample_orange = [19.58, 16.55, 10.01, 22.77, 92.17, 7.02, 110.47]
sample_papaya = [49.88, 59.05, 50.04, 33.72, 92.4, 6.74, 142.63]
sample_pigeonpeas = [20.73, 67.73, 20.29, 27.74, 48.06, 5.79, 149.46]
sample_pomegranate = [18.87, 18.75, 40.21, 21.84, 90.13, 6.43, 107.53]
sample_rice = [79.89, 47.58, 39.87, 23.69, 82.27, 6.43, 236.18]
sample_watermelon = [99.42, 17.0, 50.22, 25.59, 85.16, 6.5, 50.79]


def predict_all_crops(model, crop_code):
    samples = {
        "apple": sample_apple,
        "banana": sample_banana,
        "blackgram": sample_blackgram,
        "chickpea": sample_chickpea,
        "coconut": sample_coconut,
        "coffee": sample_coffee,
        "cotton": sample_cotton,
        "grapes": sample_grapes,
        "jute": sample_jute,
        "kidneybeans": sample_kidneybeans,
        "lentil": sample_lentil,
        "maize": sample_maize,
        "mango": sample_mango,
        "mothbeans": sample_mothbeans,
        "mungbean": sample_mungbean,
        "muskmelon": sample_muskmelon,
        "orange": sample_orange,
        "papaya": sample_papaya,
        "pigeonpeas": sample_pigeonpeas,
        "pomegranate": sample_pomegranate,
        "rice": sample_rice,
        "watermelon": sample_watermelon
    }

    for name, sample in samples.items():
        predicted_crop = predict_crop(model, crop_code, sample)
        print(f"Cultivo recomendado para {name}: {predicted_crop}")

predict_all_crops(model, crop_code)

print("\n Guardando el mejor modelo...")

# Guardar el último modelo entrenado (puedes modificar para guardar el mejor)
import pickle

model_data = {
    'model_state_dict': model.state_dict(),
    'model_architecture': model,
    'crop_encoder': crop_code
}

with open('modelo_crop.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Modelo guardado exitosamente en modelo_crop.pkl")
print("Archivo creado:", os.path.abspath('modelo_crop.pkl'))


