# IOP Estimation

## 📌 Project Overview
**IOP Estimation** (Intraocular Pressure Estimation) is a machine learning-based project designed to estimate intraocular pressure, which is a crucial factor in diagnosing and managing **glaucoma**. This project utilizes sensor data, machine learning models, and data analytics to provide insights into eye health.

## 🔬 Features
- **Real-time sensor data processing** 📊
- **Machine learning model for IOP prediction** 🤖
- **Glaucoma dataset for model training** 🏥
- **Web API for integration with healthcare applications** 🌐
- **Data visualization for insights** 📈

## 🏗️ Tech Stack
- **Arduino** (for sensor data collection)
- **Python** (data processing, ML model training)
- **Flask** (backend API)
- **Pandas, NumPy, SciKit-Learn** (data analysis and machine learning)
- **Matplotlib, Seaborn** (data visualization)

## 🚀 Installation & Setup
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/BaibhavSureka/IOP-Estimation.git
cd IOP-Estimation
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️⃣ Run the Application**
```sh
python server.py
python app.py 
```

## 📁 Project Structure
```
IOP-Estimation/
│── Ardunino.ino          # Sensor data collection script
│── app.py                # Web application backend
│── glaucoma_dataset.csv   # Dataset for training ML model
│── ml_model.py           # Machine learning model script
│── sensor_data.csv        # Collected sensor data
│── server.py             # API server for predictions
└── README.md             # Project documentation
```

## 📊 Data & Model
- **Dataset**: The project uses a glaucoma dataset containing IOP measurements and other relevant eye health indicators.
- **Model**: A trained **machine learning model (glaucoma_model.pkl)** predicts intraocular pressure based on sensor data.

## 🏥 Use Cases
- **Glaucoma Screening & Monitoring** 🔬
- **Integration with Smart Medical Devices** 📟
- **Data-Driven Healthcare Insights** 📉

## 🤝 Contributing
Feel free to contribute! If you find issues or want to add features, submit a pull request.

## 🛠️ Future Improvements
- Improve model accuracy with larger datasets 📊
- Integrate real-time cloud storage ☁️
- Deploy as a web app for accessibility 🌍

## 📩 Contact
📧 **Baibhav Sureka** - [GitHub](https://github.com/BaibhavSureka) | [LinkedIn](https://linkedin.com/in/baibhavsureka)

