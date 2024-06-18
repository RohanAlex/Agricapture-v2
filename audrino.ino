#include <ESP8266WiFi.h>
#include <FirebaseArduino.h>
#include <ESP8266WebServer.h> // Add this line
#include "DHT.h"
#include <Wire.h> 
#include <LiquidCrystal_I2C.h>
#include <EEPROM.h> // Add this line

#define FIREBASE_HOST "realtime-iot-3315a-default-rtdb.asia-southeast1.firebasedatabase.app"
#define DHTPIN D3
#define DHTTYPE DHT11
#define EEPROM_SIZE 100 // Size of EEPROM to store WiFi credentials
#define MAX_WIFI_CONNECT_ATTEMPTS 20 // Maximum number of attempts to connect to WiFi

DHT dht(DHTPIN, DHTTYPE);

const int soilPin = A0; // GPIO pin for soil moisture sensor
const int MAX_READINGS = 5; // Maximum number of moisture readings to store per month
float moistureReadings[MAX_READINGS]; // Array to store moisture readings

// LCD parameters
const int LCD_COLS = 20;
const int LCD_ROWS = 4;
LiquidCrystal_I2C lcd(0x27, LCD_COLS, LCD_ROWS);

String ssid = ""; // Variable to store WiFi SSID
String password = ""; // Variable to store WiFi password

ESP8266WebServer server(80);

// HTML form for WiFi configuration
const char* configPage = R"(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WiFi Configuration</title>
  <style>
    body {
      background-color: #f3f4f6;
      font-family: Arial, sans-serif;
    }
    .container {
      max-width: 400px;
      margin: 0 auto;
      padding-top: 20px;
    }
    .form-group {
      margin-bottom: 15px;
    }
    label {
      display: block;
      font-weight: bold;
    }
    input[type="text"],
    input[type="password"] {
      width: 100%;
      padding: 8px;
      border: 1px solid #ccc;
      border-radius: 4px;
    }
    input[type="submit"] {
      background-color: #007bff;
      color: #fff;
      border: none;
      padding: 10px 20px;
      border-radius: 4px;
      cursor: pointer;
    }
    input[type="submit"]:hover {
      background-color: #0056b3;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>WiFi Configuration</h2>
    <form action="/config" method="post">
      <div class="form-group">
        <label for="ssid">WiFi SSID:</label>
        <input type="text" id="ssid" name="ssid">
      </div>
      <div class="form-group">
        <label for="password">Password:</label>
        <input type="password" id="password" name="password">
      </div>
      <div class="form-group">
        <input type="submit" value="Submit">
      </div>
    </form>
  </div>
</body>
</html>
)";


void handleRoot() {
  server.send(200, "text/html", configPage);
}

void handleConfig() {
  String newSsid = server.arg("ssid");
  String newPassword = server.arg("password");
  
  ssid = newSsid;
  password = newPassword;

  // Store WiFi credentials in EEPROM
  for (int i = 0; i < ssid.length(); ++i) {
    EEPROM.write(i, ssid[i]);
  }
  EEPROM.write(ssid.length(), '\0');
  for (int i = 0; i < password.length(); ++i) {
    EEPROM.write(ssid.length() + 1 + i, password[i]);
  }
  EEPROM.write(ssid.length() + 1 + password.length(), '\0');
  EEPROM.commit();
  
  Serial.println("Configuration successful!");
  server.send(200, "text/plain", "Configuration successful!");
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Initializing...");
  
  // Read WiFi credentials from EEPROM
  String storedSsid = "";
  String storedPassword = "";
  for (int i = 0; i < EEPROM_SIZE; ++i) {
    char c = EEPROM.read(i);
    if (c == '\0') {
      if (storedSsid.length() == 0) {
        storedSsid = storedPassword;
      } else {
        storedPassword = "";
      }
    } else {
      if (storedSsid.length() == 0) {
        storedSsid += c;
      } else {
        storedPassword += c;
      }
    }
  }
  ssid = storedSsid;
  password = storedPassword;
  
  // Create access point to serve configuration web page
  WiFi.softAP("ESP8266AP");
  IPAddress ip = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(ip);

  // Initialize Firebase
  Firebase.begin(FIREBASE_HOST);

  // Display access point IP on LCD
  lcd.clear();
  lcd.print("AP IP: ");
  lcd.setCursor(0, 1);
  lcd.print(ip);

  // Attempt to connect to WiFi using stored credentials
  Serial.println("Attempting to connect to WiFi...");
  Serial.println("SSID: " + ssid);
  Serial.println("Password: " + password);
  WiFi.begin(ssid.c_str(), password.c_str());
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < MAX_WIFI_CONNECT_ATTEMPTS) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("");
    Serial.println("WiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());

    // Display WiFi status on LCD
    lcd.clear();
    lcd.print("Connected to WiFi");
    lcd.setCursor(0, 1);
    lcd.print("IP: ");
    lcd.print(WiFi.localIP());
  } else {
    Serial.println("");
    Serial.println("Failed to connect to WiFi");
    Serial.println("Starting in AP mode for configuration");
  }

  // Handle HTTP requests
  server.on("/", HTTP_GET, handleRoot);
  server.on("/config", HTTP_POST, handleConfig);
  server.begin();
}

void loop() {
  server.handleClient();

  // Proceed with sensor readings and Firebase updates
  float h = dht.readHumidity();
  float t = dht.readTemperature(); // Reading temperature in Celsius
  int moistureValue = analogRead(soilPin);
  float moisturePercentage = map(moistureValue, 0, 1023, 100, 0); // Map analog value to percentage
  
  // Display sensor readings on LCD line by line
  lcd.clear();
  lcd.setCursor(0, 0);
  WiFi.begin(ssid.c_str(), password.c_str());
  if (WiFi.status() == WL_CONNECTED) {
    lcd.print("WiFi Connected");
  } else {
    lcd.print("WiFi Disconnected");
  }

  lcd.setCursor(0, 1);
  lcd.print("Temp: ");
  lcd.print(t);
  lcd.print("C");
  Firebase.setFloat("SoilMoisture", moisturePercentage);

  delay(1000); // Delay before printing the next line
  
  lcd.setCursor(0, 1);
  lcd.print("Humidity: ");
  lcd.print(h);
  lcd.print("%");
  Firebase.setFloat("SoilMoisture", moisturePercentage);

  delay(1000); // Delay before printing the next line

  lcd.setCursor(0, 1);
  lcd.print("Moisture: ");
  lcd.print(moisturePercentage);
  lcd.print("%");

  // Update Firebase with temperature, humidity, and soil moisture data
  Firebase.setFloat("Temperature", t);
  Firebase.setFloat("Humidity", h);
  Firebase.setFloat("SoilMoisture", moisturePercentage);

  delay(1000); // Delay before printing the next line
}