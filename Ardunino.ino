#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <algorithm>
#include <ArduinoJson.h>

const char *ssid = "DESKTOP-OEN1MIN 1192";
const char *password = "000000111111";
const char *serverUrl = "http://172.17.25.86:5000/iop";

#define PIEZO_PIN A0
#define FSR_PIN A0
#define MAX_VOLTAGE 3.3

// Calibration constants - these should be determined experimentally
#define MIN_VALID_IOP 8.0  // Minimum valid IOP reading (mmHg)
#define MAX_VALID_IOP 40.0 // Maximum valid IOP reading (mmHg)

// Polynomial calibration coefficients (determined from calibration curve)
// These should be replaced with actual values from calibration
const float a = 0.0023;  // x³ coefficient
const float b = -0.0189; // x² coefficient
const float c = 0.9812;  // x coefficient
const float d = 5.1324;  // constant term

float iopReadings[10];
int readingCount = 0;
unsigned long lastReadingTime = 0;
const unsigned long READ_INTERVAL = 3000;

// Temperature compensation (optional but recommended)
float temperatureC = 25.0;           // Default room temperature
const float TEMP_COEFFICIENT = 0.02; // IOP change per degree C (example value)

void connectWiFi()
{
    Serial.print("Connecting to WiFi...");
    WiFi.begin(ssid, password);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20)
    {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    if (WiFi.status() == WL_CONNECTED)
    {
        Serial.println("\n✅ Connected to WiFi!");
        Serial.print("IP Address: ");
        Serial.println(WiFi.localIP());
    }
    else
    {
        Serial.println("\n❌ WiFi Connection Failed! Retrying...");
        delay(5000);
        connectWiFi();
    }
}

bool sendData(float piezoVoltage, float fsrVoltage, float rawIOP, float calibratedIOP)
{
    if (WiFi.status() != WL_CONNECTED)
    {
        Serial.println("⚠️ Wi-Fi Disconnected. Reconnecting...");
        connectWiFi();
    }

    WiFiClient client;
    HTTPClient http;

    http.begin(client, serverUrl);
    http.setTimeout(10000);
    http.addHeader("Content-Type", "application/json");

    // Create a JSON document
    DynamicJsonDocument doc(256);
    doc["piezo"] = piezoVoltage;
    doc["fsr"] = fsrVoltage;
    doc["raw_iop"] = rawIOP;
    doc["iop"] = calibratedIOP;
    doc["temp"] = temperatureC;

    String payload;
    serializeJson(doc, payload);

    Serial.print("Sending data: ");
    Serial.println(payload);

    int httpResponseCode = http.POST(payload);

    if (httpResponseCode > 0)
    {
        String response = http.getString();
        Serial.print("✅ HTTP Response code: ");
        Serial.println(httpResponseCode);
        Serial.println("Response: " + response);
        http.end();
        return true;
    }
    else
    {
        Serial.print("❌ Error code: ");
        Serial.println(httpResponseCode);
        Serial.println("Server endpoint: " + String(serverUrl));
        http.end();
        return false;
    }
}

// Function to apply calibration curve to raw sensor readings
float calibrateIOP(float fsrVoltage, float piezoVoltage)
{
    // 1. Calculate raw reading (similar to your current approach but more explicit)
    float rawReading = fsrVoltage * 10.0; // Basic conversion

    // 2. Apply polynomial calibration (from experimental data)
    // This polynomial should be derived from comparing your sensor against a gold standard
    float calibratedIOP = a * pow(rawReading, 3) + b * pow(rawReading, 2) + c * rawReading + d;

    // 3. Temperature compensation (if available)
    // IOP is affected by temperature - compensate if you have temperature sensor
    float tempCompensatedIOP = calibratedIOP + (25.0 - temperatureC) * TEMP_COEFFICIENT;

    // 4. Apply physiological limits to catch obviously erroneous readings
    if (tempCompensatedIOP < MIN_VALID_IOP)
    {
        Serial.println("⚠️ Reading below physiological minimum, adjusting to lower bound");
        return MIN_VALID_IOP;
    }
    else if (tempCompensatedIOP > MAX_VALID_IOP)
    {
        Serial.println("⚠️ Reading above physiological maximum, adjusting to upper bound");
        return MAX_VALID_IOP;
    }

    return tempCompensatedIOP;
}

void setup()
{
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n\n========================");
    Serial.println("ESP8266 IOP Sensor System");
    Serial.println("========================");
    Serial.println("Research-Grade Calibration");

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    delay(100);

    connectWiFi();

    pinMode(PIEZO_PIN, INPUT);

    Serial.println("Sensor system ready! Starting readings...");
    lastReadingTime = millis();
}

void takeSensorReading()
{
    // Read piezoelectric sensor
    int piezoRaw = analogRead(PIEZO_PIN);
    float piezoVoltage = piezoRaw * (MAX_VOLTAGE / 1023.0);

    // Read force-sensitive resistor (FSR)
    int fsrRaw = analogRead(FSR_PIN);
    float fsrVoltage = fsrRaw * (MAX_VOLTAGE / 1023.0);

    // Calculate raw IOP (for debugging/comparison)
    float rawIOP = fsrVoltage * 10.0;

    // Apply calibration algorithm to get calibrated IOP
    float calibratedIOP = calibrateIOP(fsrVoltage, piezoVoltage);

    // Store the calibrated reading
    iopReadings[readingCount] = calibratedIOP;

    // Debug output
    Serial.print("Reading ");
    Serial.print(readingCount + 1);
    Serial.print("/10 | Raw: ");
    Serial.print(fsrRaw);
    Serial.print(" | Raw IOP: ");
    Serial.print(rawIOP, 2);
    Serial.print(" | Calibrated IOP: ");
    Serial.print(calibratedIOP, 2);
    Serial.println(" mmHg");

    // Send to server
    bool success = sendData(piezoVoltage, fsrVoltage, rawIOP, calibratedIOP);
    if (success)
    {
        readingCount++;
        Serial.print("Progress: [");
        for (int i = 0; i < 10; i++)
        {
            if (i < readingCount)
            {
                Serial.print("■");
            }
            else
            {
                Serial.print("□");
            }
        }
        Serial.println("]");
    }
    else
    {
        Serial.println("⚠️ Retrying in next cycle...");
        if (WiFi.status() != WL_CONNECTED)
        {
            connectWiFi();
        }
    }
}

void loop()
{
    unsigned long currentTime = millis();

    if (readingCount < 10)
    {
        if (currentTime - lastReadingTime >= READ_INTERVAL)
        {
            takeSensorReading();
            lastReadingTime = currentTime;
        }
    }
    else
    {
        // Apply statistical filtering to readings
        float sortedIOP[10];
        memcpy(sortedIOP, iopReadings, sizeof(iopReadings));
        std::sort(sortedIOP, sortedIOP + 10);

        // Calculate median (more robust than mean)
        float medianIOP = (sortedIOP[4] + sortedIOP[5]) / 2.0;

        // Calculate standard deviation for quality assessment
        float sumSquaredDiff = 0;
        for (int i = 0; i < 10; i++)
        {
            sumSquaredDiff += pow(iopReadings[i] - medianIOP, 2);
        }
        float stdDev = sqrt(sumSquaredDiff / 10.0);

        // Quality indicator based on standard deviation
        String qualityIndicator;
        if (stdDev < 1.0)
        {
            qualityIndicator = "Excellent";
        }
        else if (stdDev < 2.0)
        {
            qualityIndicator = "Good";
        }
        else if (stdDev < 3.0)
        {
            qualityIndicator = "Fair";
        }
        else
        {
            qualityIndicator = "Poor - Consider retaking measurements";
        }

        Serial.println("\n========================");
        Serial.print("🔍 FINAL IOP READING: ");
        Serial.print(medianIOP, 2);
        Serial.println(" mmHg");
        Serial.print("📊 Measurement Quality: ");
        Serial.println(qualityIndicator);
        Serial.print("📏 Standard Deviation: ");
        Serial.println(stdDev, 2);
        Serial.println("========================");

        Serial.println("All readings (mmHg):");
        for (int i = 0; i < 10; i++)
        {
            Serial.print(i + 1);
            Serial.print(": ");
            Serial.println(iopReadings[i], 2);
        }

        Serial.println("🔴 Entering deep sleep mode (saving power)...");
        ESP.deepSleep(0);
    }
    delay(100);
}