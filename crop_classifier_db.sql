-- DB Crops

SET foreign_key_checks = 0;
DROP DATABASE IF EXISTS crop_classifier_db;
CREATE DATABASE crop_classifier_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE crop_classifier_db;

-- =====================================================
-- TABLA DE USUARIOS
-- =====================================================
CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    location VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_username (username),
    INDEX idx_email (email)
) ENGINE=InnoDB;

-- =====================================================
-- TABLA DE CULTIVOS
-- =====================================================
CREATE TABLE crops (
    crop_id INT AUTO_INCREMENT PRIMARY KEY,
    crop_name VARCHAR(50) NOT NULL UNIQUE,
    crop_label INT NOT NULL UNIQUE COMMENT 'Label para modelo ML (0-21)',
    scientific_name VARCHAR(100),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_crop_name (crop_name),
    INDEX idx_crop_label (crop_label)
) ENGINE=InnoDB;

-- =====================================================
-- TABLA DE DATOS CLIMÁTICOS DEL USUARIO
-- =====================================================
CREATE TABLE user_climate_data (
    data_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    
    -- Variables del modelo ML 
    nitrogen DECIMAL(8,2) NOT NULL COMMENT 'N - Nitrógeno',
    phosphorus DECIMAL(8,2) NOT NULL COMMENT 'P - Fósforo',
    potassium DECIMAL(8,2) NOT NULL COMMENT 'K - Potasio',
    temperature DECIMAL(5,2) NOT NULL COMMENT 'Temperatura °C',
    humidity DECIMAL(5,2) NOT NULL COMMENT 'Humedad %',
    ph_level DECIMAL(4,2) NOT NULL COMMENT 'pH del suelo',
    rainfall DECIMAL(8,2) NOT NULL COMMENT 'Precipitación mm',
    
    location VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    
    INDEX idx_user_climate_user_id (user_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB;

-- =====================================================
-- TABLA DE PREDICCIONES
-- =====================================================
CREATE TABLE crop_predictions (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    data_id INT NOT NULL,
    user_id INT NOT NULL,
    predicted_crop_id INT NOT NULL,
    
    confidence_score DECIMAL(5,4) NOT NULL COMMENT 'Confianza 0.0-1.0',
    model_architecture VARCHAR(20) DEFAULT 'deep' COMMENT 'simple/deep/dropout',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (data_id) REFERENCES user_climate_data(data_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (predicted_crop_id) REFERENCES crops(crop_id),
    
    UNIQUE KEY unique_prediction_per_data (data_id),
    
    INDEX idx_predictions_user_id (user_id),
    INDEX idx_predictions_crop_id (predicted_crop_id),
    INDEX idx_confidence_score (confidence_score DESC)
) ENGINE=InnoDB;

-- =====================================================
-- INSERTAR DATOS INICIALES DE CULTIVOS
-- =====================================================

INSERT INTO crops (crop_label, crop_name, scientific_name, description) VALUES
(0, 'apple', 'Malus domestica', 'Temperate fruit tree'),
(1, 'banana', 'Musa acuminata', 'Tropical fruit crop'),
(2, 'blackgram', 'Vigna mungo', 'Black lentil crop'),
(3, 'chickpea', 'Cicer arietinum', 'Legume crop rich in protein'),
(4, 'coconut', 'Cocos nucifera', 'Palm tree with multiple uses'),
(5, 'coffee', 'Coffea arabica', 'Bean crop for beverage production'),
(6, 'cotton', 'Gossypium hirsutum', 'Fiber crop for textile industry'),
(7, 'grapes', 'Vitis vinifera', 'Fruit crop for wine and fresh consumption'),
(8, 'jute', 'Corchorus capsularis', 'Fiber crop for burlap and rope'),
(9, 'kidneybeans', 'Phaseolus vulgaris', 'Common bean variety'),
(10, 'lentil', 'Lens culinaris', 'Protein-rich pulse crop'),
(11, 'maize', 'Zea mays', 'Corn crop, widely adaptable cereal'),
(12, 'mango', 'Mangifera indica', 'Tropical fruit tree'),
(13, 'mothbeans', 'Vigna aconitifolia', 'Drought-tolerant legume'),
(14, 'mungbean', 'Vigna radiata', 'Green gram legume'),
(15, 'muskmelon', 'Cucumis melo', 'Sweet melon variety'),
(16, 'orange', 'Citrus sinensis', 'Citrus fruit tree'),
(17, 'papaya', 'Carica papaya', 'Tropical fruit with enzymes'),
(18, 'pigeonpeas', 'Cajanus cajan', 'Drought-resistant legume'),
(19, 'pomegranate', 'Punica granatum', 'Antioxidant-rich fruit'),
(20, 'rice', 'Oryza sativa', 'Cereal grain crop requiring flooded fields'),
(21, 'watermelon', 'Citrullus lanatus', 'Large fruit with high water content');

-- =====================================================
-- VISTAS ÚTILES
-- =====================================================

-- Vista para consultas de predicciones con detalles
CREATE VIEW v_prediction_details AS
SELECT 
    cp.prediction_id,
    u.username,
    u.email,
    c.crop_name as predicted_crop,
    c.scientific_name,
    cp.confidence_score,
    cp.model_architecture,
    cp.created_at as prediction_date,
    
    -- Datos climáticos
    ucd.nitrogen,
    ucd.phosphorus,
    ucd.potassium,
    ucd.temperature,
    ucd.humidity,
    ucd.ph_level,
    ucd.rainfall,
    ucd.location as input_location
    
FROM crop_predictions cp
JOIN users u ON cp.user_id = u.user_id
JOIN crops c ON cp.predicted_crop_id = c.crop_id
JOIN user_climate_data ucd ON cp.data_id = ucd.data_id
ORDER BY cp.created_at DESC;

-- Vista para estadísticas por usuario
CREATE VIEW v_user_stats AS
SELECT 
    u.user_id,
    u.username,
    COUNT(cp.prediction_id) as total_predictions,
    AVG(cp.confidence_score) as avg_confidence,
    MAX(cp.created_at) as last_prediction,
    
    -- Cultivo más predicho
    (SELECT c.crop_name 
     FROM crop_predictions cp2 
     JOIN crops c ON cp2.predicted_crop_id = c.crop_id 
     WHERE cp2.user_id = u.user_id 
     GROUP BY c.crop_name 
     ORDER BY COUNT(*) DESC 
     LIMIT 1) as favorite_crop
     
FROM users u
LEFT JOIN crop_predictions cp ON u.user_id = cp.user_id
GROUP BY u.user_id, u.username;

-- Habilitar foreign keys
SET foreign_key_checks = 1;