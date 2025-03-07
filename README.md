# IOP Estimation - Portable Glaucoma Monitoring Device

## 📋 Project Overview

The IOP Estimation project develops a portable, non-contact device for measuring intraocular pressure (IOP) in glaucoma patients for at-home use. Glaucoma, a leading cause of irreversible blindness, requires regular IOP monitoring for effective management. Our solution addresses limitations of traditional monitoring methods by providing an affordable, comfortable alternative.

![WhatsApp Image 2025-03-07 at 15 20 25_fc8c977d](https://github.com/user-attachments/assets/4960271f-8595-4344-8c75-0b2286c47332)

![image](https://github.com/user-attachments/assets/f5a09484-5858-4b38-a2f4-2e49f676aae5)

## 💡 Key Features

- **Non-contact measurement** through gentle eyelid pressure
- **Piezoelectric sensor-based** corneal response detection
- **Machine learning-powered** IOP estimation
- **Real-time monitoring** with immediate feedback
- **IoT connectivity** for remote healthcare integration
- **Cost-effective alternative** to clinical tonometry

## 🔬 Sensor Technology

Our device utilizes two primary sensors:

1. **Piezoelectric Sensor**
   - Captures corneal deflection signals
   - Converts mechanical strain to electrical output
   - High sensitivity for precise measurements
   
2. **Force-Sensitive Resistor (FSR)**
   - Measures applied pressure on the eyelid
   - Converts force to variable resistance
   - Enables calibration of measurements

Together, these sensors create a comprehensive measurement system that correlates applied force with corneal response to estimate IOP levels accurately.

## 🛠️ Hardware Components

- **ESP8266 Microcontroller**: Handles data acquisition and transmission
- **Piezoelectric Sensor**: Captures pressure signals from corneal response
- **Force Sensitive Resistor**: Measures applied force
- **Breadboard GL No. 12**: Used for circuit assembly
- **LEDs and Resistors**: Visual indicators and current control
- **Bluetooth/Wi-Fi Module**: Enables wireless data transmission

## 💻 Tech Stack

- **Arduino (C++)**: Sensor data collection and hardware control
- **Python**: Data processing, machine learning model training
- **Flask**: Backend API server
- **Libraries**:
  - Pandas, NumPy, Scikit-learn: Data analysis and ML
  - Imbalanced-learn: Handling class imbalance
  - Matplotlib, Seaborn: Data visualization
  - ESP8266WiFi, ESP8266HTTPClient: IoT connectivity

## 📊 Machine Learning Model

Our system implements a K-Nearest Neighbors (KNN) regression model with the following specifications:

- **Algorithm**: KNN Regression (k=7)
- **Feature Engineering**: Age_IOP, IOP_squared, IOP_sqrt, and more
- **Resampling**: SMOTEENN for class balance
- **Model Ensemble**: Voting Soft classifier (0.8566 accuracy)
- **Cross-Validation**: 5-fold stratified CV
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, AUC

## 📁 Project Structure

```
IOP-Estimation/
├── Arduino.ino              # Firmware for ESP8266 (sensor data collection)
├── app.py                   # Machine learning model implementation
├── glaucoma_dataset.csv     # Dataset for training ML model
├── ml_model.py              # ML Model 
├── sensor_data.csv          # Collected sensor data storage
├── server.py                # Flask API server for predictions
└── README.md                # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Arduino IDE
- Python 3.7+
- Required hardware components

### Hardware Setup
1. Connect the piezoelectric sensor and FSR to the ESP8266
2. Assemble the circuit according to the schematic diagram
3. Upload the Arduino.ino firmware to the ESP8266

### Software Setup
1. Clone the repository
   ```sh
   git clone https://github.com/BaibhavSureka/IOP-Estimation.git
   cd IOP-Estimation
   ```

2. Install Python dependencies
   ```sh
   pip install -r requirements.txt
   ```

3. Run the application
   ```sh
   python server.py
   python app.py
   ```

## 📊 Data Flow

1. User applies gentle pressure to closed eyelid
2. FSR measures applied force and converts to electrical signal
3. Piezoelectric sensor detects corneal deflection
4. ESP8266 processes raw data and applies calibration
5. KNN algorithm estimates IOP based on sensor readings
6. Results display on device screen and transmit to mobile app
7. Data stored for tracking and analysis

## 🏥 Clinical Significance

Compared to traditional Goldmann Applanation Tonometry (GAT), our approach offers:
- No corneal contact requirement
- No anesthesia needed
- Portable, at-home use
- Continuous monitoring vs. single-point measurements
- Detection of IOP fluctuations (key risk factor)
- 0.8566 accuracy approaching clinical standards

## 📞 Contact

For questions or collaborations, please contact:
- 📧 **Baibhav Sureka**: [GitHub](https://github.com/BaibhavSureka) | [LinkedIn](https://linkedin.com/in/baibhavsureka)

## 📚 References

1. Che Hamzah, J., Daka, Q. & Azuara-Blanco, A. Home monitoring for glaucoma. Eye 34, 155–160 (2020).
2. Wu, K.Y., et al. Advancements in Wearable and Implantable Intraocular Pressure Biosensors for Ophthalmology: A Comprehensive Review. Micromachines, 14(10), 1915 (2023).
3. Daka, Q. MD, PhD, et al. Home-Based Perimetry for Glaucoma: Where Are We Now? Journal of Glaucoma 31(6): p 361-374 (2022).

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
