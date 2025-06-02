# 🌾 Crop Classifier Web App (Cloud-Based)

This project is a full-stack web application that allows users to receive crop recommendations based on soil and environmental conditions. The solution is powered by a trained neural network model deployed in our university's cloud infrastructure.

Users interact through a web interface, and their input is processed in the backend via an API, which returns the most suitable crop to plant.

---

## 🚀 Project Goals

- Deploy a machine learning crop classifier in the cloud.
- Build a user-friendly frontend for data input.
- Connect frontend and backend through a RESTful API.
- Support users with intelligent crop recommendations.

---

## 📊 Model Overview

- Neural network trained on a dataset of 2200+ crop records.
- Inputs: `N`, `P`, `K`, `temperature`, `humidity`, `pH`, `rainfall`.
- Outputs: Crop label prediction (22 classes).
- Loss function: CrossEntropyLoss
- Optimizer: Adam

---