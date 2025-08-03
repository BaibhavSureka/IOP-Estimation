#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <SoftwareSerial.h>

// Connect Arduino TX -> ESP8266 D6 (GPIO12)
// Connect GND to GND
SoftwareSerial softSerial(D6, D5); // RX, TX (we only use RX)

const char *ssid = "Baibhav";
const char *password = "9693354356";
const char *serverUrl = "http://192.168.151.175:5000/iop";

void connectWiFi()
{
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20)
    {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    if (WiFi.status() == WL_CONNECTED)
    {
        Serial.println("\n✅ WiFi Connected!");
        Serial.println(WiFi.localIP());
    }
    else
    {
        Serial.println("\n❌ WiFi Connection Failed. Retrying...");
        delay(5000);
        connectWiFi();
    }
}

void setup()
{
    Serial.begin(115200);   // Debug monitor
    softSerial.begin(9600); // Talking to Arduino UNO
    WiFi.mode(WIFI_STA);
    connectWiFi();
}

void loop()
{
    if (softSerial.available())
    {
        String data = softSerial.readStringUntil('\n');
        int sep1 = data.indexOf('|');
        int sep2 = data.lastIndexOf('|');

        if (sep1 > 0 && sep2 > sep1)
        {
            float piezoVoltage = data.substring(0, sep1).toFloat();
            float fsrVoltage = data.substring(sep1 + 1, sep2).toFloat();
            float iopValue = data.substring(sep2 + 1).toFloat();

            if (WiFi.status() != WL_CONNECTED)
            {
                connectWiFi();
            }

            WiFiClient client;
            HTTPClient http;
            http.begin(client, serverUrl);
            http.addHeader("Content-Type", "application/json");

            String payload = "{\"piezo\":" + String(piezoVoltage, 3) +
                             ",\"fsr\":" + String(fsrVoltage, 3) +
                             ",\"iop\":" + String(iopValue, 2) + "}";

            Serial.println("Sending: " + payload);
            int responseCode = http.POST(payload);

            if (responseCode > 0)
            {
                Serial.print("✅ Sent | Server Response: ");
                Serial.println(responseCode);
                String response = http.getString();
                Serial.println("Response: " + response);
            }
            else
            {
                Serial.print("❌ Failed to send. Error code: ");
                Serial.println(responseCode);
            }
            http.end();
        }
    }
}