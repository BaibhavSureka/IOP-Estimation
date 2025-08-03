#define PIEZO_PIN A0
#define FSR_PIN A1

#define PIEZO_SENSITIVITY 0.75
#define FSR_SENSITIVITY 15.0
#define AREA_CONSTANT 1.2
#define IOP_SCALING 15.0

void setup()
{
    Serial.begin(9600); // Match NodeMCU baud rate
}

void loop()
{
    // Read and convert Piezo sensor
    int piezoRaw = analogRead(PIEZO_PIN);
    float piezoVoltage = piezoRaw * (5.0 / 1023.0);
    float deflection = piezoVoltage * PIEZO_SENSITIVITY;

    // Read and convert FSR sensor
    int fsrRaw = analogRead(FSR_PIN);
    float fsrVoltage = fsrRaw * (5.0 / 1023.0);
    float appliedForce = fsrVoltage * FSR_SENSITIVITY;

    // Calculate IOP
    float IOP = (appliedForce / AREA_CONSTANT) * IOP_SCALING;
    if (IOP > 50)
        IOP = 25.0;

    // Send data over Serial in format: piezo|fsr|iop
    Serial.print(piezoVoltage, 3);
    Serial.print("|");
    Serial.print(fsrVoltage, 3);
    Serial.print("|");
    Serial.println(IOP, 2);

    delay(2000); // 2 second interval
}