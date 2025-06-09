from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
import torch
import torch.nn as nn
import pickle
import numpy as np
import hashlib
from datetime import datetime
import jwt
from functools import wraps

app = Flask(__name__)
app.secret_key = 'test@byte87-' # Cambiar por una clave secreta más segura en producción 
CORS(app, origins=['http://localhost:3000'])

# Configuración de la base de datos
import os
"""
DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'port': int(os.getenv("DB_PORT", 3306)),
    'database': os.getenv("DB_NAME", "crop_classifier_db"),
    'user': os.getenv("DB_USER", "usuario_app"),
    'password': os.getenv("DB_PASSWORD", "pass_app"),
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci'
}"""
DB_CONFIG = {
    'host': 'localhost',
    'database': 'crop_classifier_db',
    'user': 'root',
    'password': '', #Cambiar
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci'
}
# Definir la clase del modelo
class NClassifier(nn.Module):
    def __init__(self, architecture):
        super(NClassifier, self).__init__()
        self.encoder = nn.Sequential(*architecture)

    def forward(self, x):
        return self.encoder(x)

def load_model():
    """Cargar modelo con manejo de errores mejorado"""
    try:
        # Intentar cargar el modelo guardado para API
        with open('modelo_crop_api.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Verificar si tiene formato viejo (con model_architecture)
        if 'model_architecture' in model_data:
            print("Cargando modelo en formato original")
            model = model_data['model_architecture']
            crop_encoder = model_data['crop_encoder']
            model.eval()
        else:
            print("Cargando modelo en formato API")
            # Formato nuevo
            architecture = model_data['architecture']
            model = NClassifier(architecture)
            model.load_state_dict(model_data['model_state_dict'])
            model.eval()
            crop_encoder = model_data['crop_encoder']
        
        print("Modelo cargado desde modelo_crop_api.pkl")
        return model, crop_encoder
        
    except FileNotFoundError:
        print("No se encontró modelo_crop_api.pkl")
        return None, None
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None, None

# Cargar modelo al iniciar
MODEL, CROP_ENCODER = load_model()

def get_db_connection():
    """Crear conexión a la base de datos"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error conectando a la BD: {e}")
        return None

def hash_password(password):
    """Hash de contraseña simple"""
    return hashlib.sha256(password.encode()).hexdigest()

def token_required(f):
    """Decorador para verificar autenticación"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token requerido'}), 401
        
        try:
            token = token.split(" ")[1] if " " in token else token
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user_id = data['user_id']
        except:
            return jsonify({'error': 'Token inválido'}), 401
        
        return f(current_user_id, *args, **kwargs)
    return decorated

@app.route("/api/auth/signin", methods=["POST"])
def register_user():
    """Registrar nuevo usuario"""
    try:
        data = request.get_json()
        required_fields = ['username', 'email', 'password']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Faltan campos requeridos'}), 400
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Error de conexión a BD'}), 500
        
        cursor = connection.cursor()
        
        # Verificar si usuario ya existe
        cursor.execute("SELECT user_id FROM users WHERE username = %s OR email = %s", 
                      (data['username'], data['email']))
        if cursor.fetchone():
            return jsonify({'error': 'Usuario o email ya existe'}), 400
        
        # Crear usuario
        password_hash = hash_password(data['password'])
        insert_query = """
            INSERT INTO users (username, email, password_hash, full_name)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            data['username'],
            data['email'], 
            password_hash,
            data.get('full_name')
        ))
        
        user_id = cursor.lastrowid
        connection.commit()
        cursor.close()
        connection.close()
        
        # Generar token
        token = jwt.encode({
            'user_id': user_id,
            'username': data['username']
        }, app.secret_key, algorithm='HS256')
        
        return jsonify({
            'message': 'Usuario creado exitosamente',
            'user_id': user_id,
            'token': token,
            'success': True
        }), 201
        
    except Error as e:
        return jsonify({'error': f'Error de BD: {str(e)}'}), 500

@app.route("/api/auth/login", methods=["POST"])
def login_user():
    """Iniciar sesión"""
    try:
        data = request.get_json()
        if not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Username y password requeridos'}), 400
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Error de conexión a BD'}), 500
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT user_id, username, password_hash, full_name 
            FROM users WHERE username = %s
        """, (data['username'],))
        
        user = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not user or user['password_hash'] != hash_password(data['password']):
            return jsonify({'error': 'Credenciales inválidas'}), 401
        
        # Generar token
        token = jwt.encode({
            'user_id': user['user_id'],
            'username': user['username']
        }, app.secret_key, algorithm='HS256')
        
        return jsonify({
            'message': 'Login exitoso',
            'user': {
                'user_id': user['user_id'],
                'username': user['username'],
                'full_name': user['full_name']
            },
            'token': token,
            'success': True
        })
        
    except Error as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
@token_required
def predict_crop(current_user_id):
    """Predicción de cultivos para usuario autenticado"""
    try:
        data = request.get_json()
        
        # Validar campos requeridos
        required_fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 
                          'humidity', 'ph_level', 'rainfall']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Faltan campos requeridos'}), 400
        
        if MODEL is None:
            return jsonify({'error': 'Modelo no disponible'}), 500
        
        # Extraer características
        features = [float(data[field]) for field in required_fields]
        
        # Hacer predicción
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = MODEL(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities.max().item()
        
        predicted_crop = CROP_ENCODER.inverse_transform([predicted_class])[0]
        
        # Guardar en base de datos
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Error de conexión a BD'}), 500
        
        cursor = connection.cursor()
        
        # Insertar datos climáticos
        climate_query = """
            INSERT INTO user_climate_data 
            (user_id, nitrogen, phosphorus, potassium, temperature, humidity, ph_level, rainfall)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(climate_query, (
            current_user_id, *features
        ))
        data_id = cursor.lastrowid
        
        # Obtener crop_id
        cursor.execute("SELECT crop_id FROM crops WHERE crop_label = %s", (predicted_class,))
        crop_result = cursor.fetchone()
        
        if crop_result:
            crop_id = crop_result[0]
            
            # Insertar predicción
            prediction_query = """
                INSERT INTO crop_predictions 
                (data_id, user_id, predicted_crop_id, confidence_score, model_architecture)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(prediction_query, (
                data_id, current_user_id, crop_id, confidence, 
                data.get('model_architecture', 'deep')
            ))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return jsonify({
            'predicted_crop': predicted_crop,
            'confidence': float(confidence),
            'crop_label': int(predicted_class),
            'data_id': data_id,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Error en predicción: {str(e)}'}), 500

@app.route('/predict-simple', methods=['POST'])
def predict_simple():
    """Predicción simple sin autenticación (para pruebas)"""
    try:
        data = request.get_json()
        
        # Validar campos requeridos (usando nombres originales del CSV)
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Faltan campos requeridos: {required_fields}'}), 400
        
        if MODEL is None:
            return jsonify({'error': 'Modelo no disponible'}), 500
        
        # Extraer características
        features = [float(data[field]) for field in required_fields]
        
        # Hacer predicción
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = MODEL(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities.max().item()
        
        predicted_crop = CROP_ENCODER.inverse_transform([predicted_class])[0]
        
        return jsonify({
            'predicted_crop': predicted_crop,
            'confidence': float(confidence),
            'crop_label': int(predicted_class),
            'input_features': features,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Error en predicción: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Verificar estado de la API"""
    model_status = "loaded" if MODEL is not None else "not_loaded"
    
    try:
        connection = get_db_connection()
        db_status = "connected" if connection else "disconnected"
        if connection:
            connection.close()
    except:
        db_status = "error"
    
    crops_available = len(CROP_ENCODER.classes_) if CROP_ENCODER else 0
    
    return jsonify({
        'status': 'healthy',
        'model': model_status,
        'database': db_status,
        'crops_available': crops_available,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/crops', methods=['GET'])
def get_all_crops():
    """Obtener lista de todos los cultivos"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Error de conexión a BD'}), 500
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT crop_id, crop_name, crop_label, scientific_name, description 
            FROM crops 
            ORDER BY crop_name
        """)
        
        crops = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return jsonify({
            'crops': crops,
            'count': len(crops),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 50)
    if MODEL is None:
        print("Modelo NO cargado")
        print("Ejecuta: python save_model_standalone.py")
    else:
        print("Modelo cargado correctamente")
        print(f"{len(CROP_ENCODER.classes_)} cultivos disponibles")
    
    try:
        connection = get_db_connection()
        if connection:
            print("Base de datos conectada")
            connection.close()
        else:
            print("Error de conexión a BD")
    except:
        print("Error de conexión a BD")
    
    print("=" * 50)
    print("Iniciando API Flask en puerto 5001...")
    
    # Usar puerto 5001 en lugar de 5000
    app.run(debug=True, host='0.0.0.0', port=5001)