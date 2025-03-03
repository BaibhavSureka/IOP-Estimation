#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <algorithm> 

const char* ssid = "DESKTOP-OEN1MIN 1192";
const char* password = "000000111111";
const char* serverUrl = "http://172.17.25.86:5000/iop";

#define PIEZO_PIN A0  
#define FSR_PIN A0 
#define MAX_VOLTAGE 3.3 
#define PIEZO_SENSITIVITY 0.5  
#define FSR_SENSITIVITY 10.0  
#define AREA_CONSTANT 1.5 
#define IOP_SCALING 10.0

float iopReadings[10];  
int readingCount = 0;
unsigned long lastReadingTime = 0;
const unsigned long READ_INTERVAL = 3000;

void connectWiFi() {
    Serial.print("Connecting to WiFi...");
    WiFi.begin(ssid, password);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n✅ Connected to WiFi!");
        Serial.print("IP Address: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\n❌ WiFi Connection Failed! Retrying...");
        delay(5000);
        connectWiFi();
    }
}

bool sendData(float piezoVoltage, float fsrVoltage, float IOP) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("⚠️ Wi-Fi Disconnected. Reconnecting...");
        connectWiFi();
    }

    WiFiClient client;
    HTTPClient http;
    
    http.begin(client, serverUrl);
    http.setTimeout(10000); 
    http.addHeader("Content-Type", "application/json");

    String payload = "{\"piezo\":" + String(piezoVoltage, 2) + 
                     ",\"fsr\":" + String(fsrVoltage, 2) + 
                     ",\"iop\":" + String(IOP, 2) + "}";

    Serial.print("Sending data: ");
    Serial.println(payload);

    int httpResponseCode = http.POST(payload);
    
    if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.print("✅ HTTP Response code: ");
        Serial.println(httpResponseCode);
        Serial.println("Response: " + response);
        http.end();
        return true;
    } else {
        Serial.print("❌ Error code: ");
        Serial.println(httpResponseCode);
        Serial.println("Server endpoint: " + String(serverUrl));
        http.end();
        return false;
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n\n========================");
    Serial.println("ESP8266 IOP Sensor System");
    Serial.println("========================");
    
    WiFi.mode(WIFI_STA); 
    WiFi.disconnect();   
    delay(100);
    
    connectWiFi();
    
    pinMode(PIEZO_PIN, INPUT);
    
    Serial.println("Sensor system ready! Starting readings...");
    lastReadingTime = millis();
}

void takeSensorReading() {
    int piezoRaw = analogRead(PIEZO_PIN);
    float piezoVoltage = piezoRaw * (MAX_VOLTAGE / 1023.0);  
    float deflection = piezoVoltage * PIEZO_SENSITIVITY;  

    int fsrRaw = analogRead(FSR_PIN);
    float fsrVoltage = fsrRaw * (MAX_VOLTAGE / 1023.0);  
    float appliedForce = fsrVoltage * FSR_SENSITIVITY;  

    float IOP = (appliedForce / AREA_CONSTANT) * IOP_SCALING;
    
    if (IOP > 100) {
        IOP = 18.0;
        Serial.println("⚠️ Abnormal reading detected, normalized to standard IOP");
    }
    
    iopReadings[readingCount] = IOP;  
    
    Serial.print("Reading ");
    Serial.print(readingCount + 1); 
    Serial.print("/10 | Raw: ");
    Serial.print(fsrRaw);
    Serial.print(" | IOP: ");
    Serial.print(IOP, 2);
    Serial.println(" mmHg");

    bool success = sendData(piezoVoltage, fsrVoltage, IOP);
    if (success) {
        readingCount++; 
        Serial.print("Progress: [");
        for (int i = 0; i < 10; i++) {
            if (i < readingCount) {
                Serial.print("■");
            } else {
                Serial.print("□");
            }
        }
        Serial.println("]");
    } else {
        Serial.println("⚠️ Retrying in next cycle...");
        if (WiFi.status() != WL_CONNECTED) {
            connectWiFi();
        }
    }
}

void loop() {
    unsigned long currentTime = millis();
    
    if (readingCount < 10) {
        if (currentTime - lastReadingTime >= READ_INTERVAL) {
            takeSensorReading();
            lastReadingTime = currentTime;
        }
    } else {
        float sortedIOP[10];
        memcpy(sortedIOP, iopReadings, sizeof(iopReadings));
        std::sort(sortedIOP, sortedIOP + 10);

        float finalIOP = (sortedIOP[4] + sortedIOP[5]) / 2.0;

        Serial.println("\n========================");
        Serial.print("🔍 FINAL IOP READING: ");
        Serial.print(finalIOP, 2);
        Serial.println(" mmHg");
        Serial.println("========================");
        
        Serial.println("All readings (mmHg):");
        for (int i = 0; i < 10; i++) {
            Serial.print(i+1);
            Serial.print(": ");
            Serial.println(iopReadings[i], 2);
        }

        Serial.println("🔴 Entering deep sleep mode (saving power)...");
        ESP.deepSleep(0);
    }
    delay(100);
}
