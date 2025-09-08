// Arduino UNO Code: Read FSR and Piezo Sensor, calculate IOP

#define FSR_PIN A0
#define PIEZO_PIN A1

// Calibration constants (adjust based on hospital feedback/lab tests)
#define FSR_VOLTAGE_TO_FORCE 15.0 // N per Volt (or adjust via testing)
#define PIEZO_VOLTAGE_TO_AREA 0.8 // cm^2 per Volt (approximate contact area from deflection)
#define IOP_SCALE_FACTOR 1.0      // To scale result to mmHg range if needed

void setup()
{
    Serial.begin(9600); // Baud rate for UART (to ESP8266 or Serial Monitor)
}

void loop()
{
    // Read analog voltage from FSR and Piezo
    int fsrRaw = analogRead(FSR_PIN);
    float fsrVoltage = fsrRaw * (5.0 / 1023.0);             // Convert to Voltage
    float appliedForce = fsrVoltage * FSR_VOLTAGE_TO_FORCE; // Force in Newtons

    int piezoRaw = analogRead(PIEZO_PIN);
    float piezoVoltage = piezoRaw * (5.0 / 1023.0);           // Convert to Voltage
    float contactArea = piezoVoltage * PIEZO_VOLTAGE_TO_AREA; // Area in cm^2

    // Avoid divide by zero
    if (contactArea < 0.01)
        contactArea = 0.01;

    // Calculate IOP in mmHg (approx)
    float IOP = (appliedForce / contactArea) * IOP_SCALE_FACTOR;

    // Clamp to physiological range
    if (IOP > 60.0)
        IOP = 60.0;
    if (IOP < 0.0)
        IOP = 0.0;

    // Print to Serial for ESP or logging
    Serial.print(piezoVoltage, 3);
    Serial.print("|");
    Serial.print(fsrVoltage, 3);
    Serial.print("|");
    Serial.println(IOP, 2);

    delay(2000); // 2 second interval
}
