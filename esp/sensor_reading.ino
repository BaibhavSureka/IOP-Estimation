#define PIEZO_PIN A0
#define FSR_PIN A1

// Replace these constants with actual calibration values
#define PIEZO_SENSITIVITY 0.75 // V per unit deflection (example)
#define FSR_SENSITIVITY 15.0   // Force conversion from voltage (example)

// For applanation area, corresponding to a 3.06 mm diameter:
// Area = π * (d/2)^2 = π * (1.53 mm)^2 ≈ 7.35 mm² = 7.35e-6 m²
#define CONTACT_AREA_MM2 7.35                     // in mm²
#define CONTACT_AREA_M2 (CONTACT_AREA_MM2 * 1e-6) // in m²

// Conversion: Pascal (N/m²) to mmHg (1 mmHg ≈ 133.322 Pa)
#define PA_TO_MMHG 0.00750062

void setup()
{
    Serial.begin(9600); // Must match NodeMCU baud rate
}

void loop()
{
    // Read sensors
    int piezoRaw = analogRead(PIEZO_PIN);
    float piezoVoltage = piezoRaw * (5.0 / 1023.0);
    // Example: convert to deflection if needed (not used directly)
    float deflection = piezoVoltage * PIEZO_SENSITIVITY;

    int fsrRaw = analogRead(FSR_PIN);
    float fsrVoltage = fsrRaw * (5.0 / 1023.0);

    // Convert voltage to force (Newtons)
    float forceN = fsrVoltage * FSR_SENSITIVITY;

    // Calculate pressure in Pascal: P = F / A
    float pressurePa = forceN / CONTACT_AREA_M2;

    // Convert to mmHg:
    float IOP_mmhg = pressurePa * PA_TO_MMHG;

    // Safety clamp
    if (IOP_mmhg > 50)
        IOP_mmhg = 25.0;

    // Send formatted sensor data
    Serial.print(piezoVoltage, 3);
    Serial.print("|");
    Serial.print(fsrVoltage, 3);
    Serial.print("|");
    Serial.println(IOP_mmhg, 2);

    delay(2000);
}
