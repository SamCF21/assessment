import mysql.connector
from mysql.connector import Error
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
import json
from datetime import datetime

class CropRecommendationMySQL:
    def __init__(self, host='localhost', database='crop_classifier_db', 
                 user='root', password=''):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.model = None
        self.crop_encoder = None
        
    def connect_db(self):
        """Conectar a la base de datos MySQL"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                charset='utf8mb4',
                collation='utf8mb4_unicode_ci'
            )
            
            if self.connection.is_connected():
                print("Conexión exitosa a MySQL")
                return True
        except Error as e:
            print(f"Error conectando a MySQL: {e}")
            return False
    
    def create_user(self, username, email, password_hash, full_name=None, location=None):
        """Crear nuevo usuario"""
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO users (username, email, password_hash, full_name, location)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (username, email, password_hash, full_name, location))
            self.connection.commit()
            user_id = cursor.lastrowid
            cursor.close()
            print(f"Usuario creado con ID: {user_id}")
            return user_id
        except Error as e:
            print(f"Error creando usuario: {e}")
            return None
    
    def save_model(self, model, crop_encoder, model_path='modelo_crop.pkl'):
        """Guardar el modelo entrenado"""
        model_data = {
            'model_state_dict': model.state_dict(),
            'model_architecture': model,
            'crop_encoder': crop_encoder
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Modelo guardado en {model_path}")
    
    def load_model(self, model_path='modelo_crop.pkl'):
        """Cargar modelo guardado"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model_architecture']
            self.model.load_state_dict(model_data['model_state_dict'])
            self.crop_encoder = model_data['crop_encoder']
            self.model.eval()
            print("Modelo cargado exitosamente")
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    def predict_crop(self, input_features):
        """Hacer predicción con el modelo cargado"""
        if self.model is None:
            print("Modelo no cargado")
            return None
        
        input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities.max().item()
        
        predicted_crop = self.crop_encoder.inverse_transform([predicted_class])[0]
        
        return {
            'crop': predicted_crop,
            'crop_label': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.numpy()
        }
    
    def insert_climate_data_and_predict(self, user_id, nitrogen, phosphorus, potassium, 
                                       temperature, humidity, ph_level, rainfall, 
                                       location=None, model_architecture='deep'):
        """Insertar datos climáticos y hacer predicción completa"""
        try:
            # Preparar características para el modelo
            features = [nitrogen, phosphorus, potassium, temperature, humidity, ph_level, rainfall]
            
            # Hacer predicción
            prediction = self.predict_crop(features)
            if not prediction:
                return None
            
            cursor = self.connection.cursor()
            
            # Insertar datos climáticos
            climate_query = """
                INSERT INTO user_climate_data 
                (user_id, nitrogen, phosphorus, potassium, temperature, humidity, ph_level, rainfall, location)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(climate_query, (user_id, nitrogen, phosphorus, potassium, 
                                         temperature, humidity, ph_level, rainfall, location))
            data_id = cursor.lastrowid
            
            # Obtener crop_id basado en el crop_label predicho
            crop_query = "SELECT crop_id FROM crops WHERE crop_label = %s"
            cursor.execute(crop_query, (prediction['crop_label'],))
            crop_result = cursor.fetchone()
            
            if not crop_result:
                print(f"Error: No se encontró crop_id para label {prediction['crop_label']}")
                return None
                
            crop_id = crop_result[0]
            
            # Insertar predicción
            prediction_query = """
                INSERT INTO crop_predictions 
                (data_id, user_id, predicted_crop_id, confidence_score, model_architecture)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(prediction_query, (data_id, user_id, crop_id, 
                                            prediction['confidence'], model_architecture))
            prediction_id = cursor.lastrowid
            
            self.connection.commit()
            cursor.close()
            
            return {
                'data_id': data_id,
                'prediction_id': prediction_id,
                'prediction': prediction
            }
            
        except Error as e:
            print(f"Error insertando datos y predicción: {e}")
            self.connection.rollback()
            return None
    
    def get_user_predictions(self, user_id, limit=10):
        """Obtener predicciones de un usuario usando la vista"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
                SELECT * FROM v_prediction_details 
                WHERE user_id = %s 
                ORDER BY prediction_date DESC 
                LIMIT %s
            """
            cursor.execute(query, (user_id, limit))
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            print(f"Error obteniendo predicciones: {e}")
            return []
    
    def get_crop_statistics(self):
        """Obtener estadísticas generales de cultivos"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
                SELECT 
                    c.crop_name,
                    COUNT(cp.prediction_id) as prediction_count,
                    AVG(cp.confidence_score) as avg_confidence,
                    MAX(cp.created_at) as last_predicted
                FROM crops c
                LEFT JOIN crop_predictions cp ON c.crop_id = cp.predicted_crop_id
                GROUP BY c.crop_id, c.crop_name
                ORDER BY prediction_count DESC
            """
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            print(f"Error obteniendo estadísticas: {e}")
            return []
    
    def get_user_stats(self, user_id):
        """Obtener estadísticas de un usuario específico"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT * FROM v_user_stats WHERE user_id = %s"
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
            cursor.close()
            return result
        except Error as e:
            print(f"Error obteniendo estadísticas de usuario: {e}")
            return None
    
    def search_users_by_location(self, location):
        """Buscar usuarios por ubicación"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
                SELECT user_id, username, full_name, location, created_at
                FROM users 
                WHERE location LIKE %s
            """
            cursor.execute(query, (f"%{location}%",))
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            print(f"Error buscando usuarios: {e}")
            return []
    
    def get_popular_crops_by_region(self, location_filter=None):
        """Obtener cultivos populares por región"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            if location_filter:
                query = """
                    SELECT 
                        c.crop_name,
                        ucd.location,
                        COUNT(cp.prediction_id) as prediction_count,
                        AVG(cp.confidence_score) as avg_confidence
                    FROM crop_predictions cp
                    JOIN crops c ON cp.predicted_crop_id = c.crop_id
                    JOIN user_climate_data ucd ON cp.data_id = ucd.data_id
                    WHERE ucd.location LIKE %s
                    GROUP BY c.crop_id, c.crop_name, ucd.location
                    ORDER BY prediction_count DESC
                """
                cursor.execute(query, (f"%{location_filter}%",))
            else:
                query = """
                    SELECT 
                        c.crop_name,
                        COUNT(cp.prediction_id) as total_predictions,
                        AVG(cp.confidence_score) as avg_confidence
                    FROM crop_predictions cp
                    JOIN crops c ON cp.predicted_crop_id = c.crop_id
                    GROUP BY c.crop_id, c.crop_name
                    ORDER BY total_predictions DESC
                    LIMIT 10
                """
                cursor.execute(query)
            
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            print(f"Error obteniendo cultivos populares: {e}")
            return []
    
    def close_connection(self):
        """Cerrar conexión"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Conexión a MySQL cerrada")

# Ejemplo de uso completo
def main():
    # Configuración de la base de datos
    db_config = {
        'host': 'localhost',
        'database': 'crop_classifier_db',
        'user': 'root', 
        'password': ''   
    }
    
    # Crear instancia
    crop_db = CropRecommendationMySQL(**db_config)
    
    # Conectar
    if crop_db.connect_db():
        # Cargar modelo (asumiendo que ya existe)
        # crop_db.load_model('modelo_crop.pkl')
        
        # Ejemplo: crear usuario
        # user_id = crop_db.create_user(
        #     username='agricultor1',
        #     email='agricultor1@example.com',
        #     password_hash='hash_seguro_aqui',
        #     full_name='Juan Pérez',
        #     location='Jalisco, México'
        # )
        
        # Ejemplo: hacer predicción completa
        # if user_id:
        #     result = crop_db.insert_climate_data_and_predict(
        #         user_id=user_id,
        #         nitrogen=90,
        #         phosphorus=42,
        #         potassium=43,
        #         temperature=20.9,
        #         humidity=82.0,
        #         ph_level=6.5,
        #         rainfall=202.9,
        #         location='Campo Norte, Jalisco'
        #     )
        #     
        #     if result:
        #         prediction = result['prediction']
        #         print(f"Cultivo recomendado: {prediction['crop']}")
        #         print(f"Confianza: {prediction['confidence']:.3f}")
        
        # Obtener estadísticas
        stats = crop_db.get_crop_statistics()
        print("Estadísticas de cultivos:")
        for stat in stats[:5]:  # Top 5
            print(f"- {stat['crop_name']}: {stat['prediction_count']} predicciones")
        
        # Cerrar conexión
        crop_db.close_connection()

if __name__ == "__main__":
    main()