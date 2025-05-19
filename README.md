Certainly! Here's an improved version of the repository description that explicitly highlights **federated learning**, especially the use of **FedMA-style aggregation**:

---

## 🌫️ Multimodal AQI Classification with Federated Learning (CNN + LSTM)

This repository explores a **federated learning** approach to classifying air quality levels using two independent data modalities:

* 📷 **Image-based model (CNN):** Learns to classify air pollution severity from environmental images.
* 📈 **Numerical model (LSTM):** Processes time-series AQI sensor data (e.g., PM2.5, PM10, NO₂) for classification.

The models are trained **separately** on local data sources, mimicking real-world decentralized clients (e.g., an IoT sensor node vs. a camera system). They are then **aggregated using a FedMA-style algorithm** to create a global model that benefits from both perspectives without sharing raw data.

---

### 🔧 Components

* `cnn_model.py` & `lstm_model.py` – Define independent client models for image and numerical data.
* `train_image_model.py` & `train_numerical_model.py` – Local training scripts for each client.
* `predict_image.py` & `predict_numerical.py` – Run inference and output predictions with confidence.
* `fedma_aggregate.py` – Aggregates CNN and LSTM models using layer-wise matching inspired by **Federated Matched Averaging (FedMA)**.
* `results.py` – Evaluates predictions and plots confusion matrices for each model.

---

### 🧠 Why Federated?

This project simulates a **federated learning environment** where:

* Clients have **non-IID** data (e.g., images vs. AQI sensor readings).
* Data **remains private** on the client side.
* Only model parameters are shared for aggregation.

By applying a **FedMA-style strategy**, the project aligns semantically similar neurons across models to build a cohesive, global classifier that reflects knowledge from diverse input modalities.

---

### 📊 Evaluation

Each model is evaluated independently:

* Confusion Matrix (per class)
* Classification Report
* Accuracy Score

---

### 📁 Data Sources

* **Image Dataset:** Structured in class-labeled folders (e.g., `e_Very_Unhealthy/`).
* **Sensor Dataset:** CSV files containing timestamped pollutant concentrations and AQI class labels.

---

### 🚀 Project Goals

* Evaluate classification performance of each modality independently.
* Simulate cross-silo federated learning with heterogenous data modalities.
* Investigate **model fusion via FedMA** to leverage complementary client knowledge.

---

Let me know if you want this adapted into a `README.md` with markdown formatting or if you'd like to include images, diagrams, or usage instructions.
