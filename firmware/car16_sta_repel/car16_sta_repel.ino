#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_idf_version.h>

#if !defined(CONFIG_IDF_TARGET_ESP32C5)
#error "This car requires ESP32-C5 and the C5 wiring in docs/ESP32_C5_WIRING.md."
#endif
#if ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(5, 4, 2)
#error "This firmware requires Arduino-ESP32 with the 5GHz band-mode API."
#endif

const char *WIFI_SSID = "YAYA";
const char *WIFI_PASSWORD = "kedayaya";
const uint8_t CAR_ID = 16;
const char *CAR_NAME = "kedaya16";
const uint16_t CONTROL_PORT = 23;
const uint16_t DISCOVERY_PORT = 4210;
const char *DISCOVERY_REQUEST = "BOID_CAR_DISCOVER";


WiFiServer server(CONTROL_PORT);
WiFiUDP discoveryUdp;
WiFiClient activeClient;
String commandBuffer;
void onWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
  if (event == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
    Serial.print("[WiFi] connected SSID=");
    Serial.print(WiFi.SSID());
    Serial.print(" IP=");
    Serial.print(WiFi.localIP());
    Serial.print(" MAC=");
    Serial.print(WiFi.macAddress());
    Serial.print(" RSSI=");
    Serial.print(WiFi.RSSI());
    Serial.print(" channel=");
    Serial.print(WiFi.channel());
    Serial.print(" band=");
    Serial.println(WiFi.getBand() == WIFI_BAND_5G ? "5GHz" : "UNEXPECTED");
  } else if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
    Serial.print("[WiFi] disconnected reason=");
    Serial.println(info.wifi_sta_disconnected.reason);
  }
}

// Waveshare ESP32-C5-WIFI6-KIT-N16R8-M: TB6612-A STBY=GPIO23; TB6612-B STBY=GPIO25.
// GPIO3/25 strap SDIO edges only; this sketch uses SPI flash boot and UART/USB upload.
// GPIO6 is battery sense; GPIO15 is reserved for PSRAM. Keep both unused.
// See docs/ESP32_C5_WIRING.md; these are GPIO numbers, not header positions.
const int A_STBY = 23;
const int M1_IN1 = 0;
const int M1_IN2 = 1;
const int M1_PWM = 3;

const int M2_IN1 = 4;
const int M2_IN2 = 5;
const int M2_PWM = 8;

const int B_STBY = 25;
const int M3_IN1 = 9;
const int M3_IN2 = 10;
const int M3_PWM = 24;

int speedVal = 150;
unsigned long lastCommandMs = 0;
unsigned long lastClientRxMs = 0;
unsigned long lastReconnectAttemptMs = 0;
unsigned long lastRssiLogMs = 0;
const unsigned long COMMAND_TIMEOUT_MS = 700;
const unsigned long CLIENT_IDLE_TIMEOUT_MS = 2500;

void motorStop(int in1, int in2, int pwmPin) {
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  ledcWrite(pwmPin, 0);
}

void motorForward(int in1, int in2, int pwmPin) {
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  ledcWrite(pwmPin, speedVal);
}

void motorBackward(int in1, int in2, int pwmPin) {
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  ledcWrite(pwmPin, speedVal);
}

void stopCar() {
  motorStop(M1_IN1, M1_IN2, M1_PWM);
  motorStop(M2_IN1, M2_IN2, M2_PWM);
  motorStop(M3_IN1, M3_IN2, M3_PWM);
}

void moveForward() {
  motorStop(M1_IN1, M1_IN2, M1_PWM);
  motorBackward(M2_IN1, M2_IN2, M2_PWM);
  motorBackward(M3_IN1, M3_IN2, M3_PWM);
}

void moveBackward() {
  motorStop(M1_IN1, M1_IN2, M1_PWM);
  motorForward(M2_IN1, M2_IN2, M2_PWM);
  motorForward(M3_IN1, M3_IN2, M3_PWM);
}

void moveLeftForward() {
  motorBackward(M1_IN1, M1_IN2, M1_PWM);
  motorStop(M2_IN1, M2_IN2, M2_PWM);
  motorBackward(M3_IN1, M3_IN2, M3_PWM);
}

void moveRightForward() {
  motorForward(M1_IN1, M1_IN2, M1_PWM);
  motorBackward(M2_IN1, M2_IN2, M2_PWM);
  motorStop(M3_IN1, M3_IN2, M3_PWM);
}

void moveLeftBackward() {
  motorBackward(M1_IN1, M1_IN2, M1_PWM);
  motorForward(M2_IN1, M2_IN2, M2_PWM);
  motorStop(M3_IN1, M3_IN2, M3_PWM);
}

void moveRightBackward() {
  motorForward(M1_IN1, M1_IN2, M1_PWM);
  motorStop(M2_IN1, M2_IN2, M2_PWM);
  motorForward(M3_IN1, M3_IN2, M3_PWM);
}

void initMotors() {
  pinMode(A_STBY, OUTPUT);
  pinMode(B_STBY, OUTPUT);
  digitalWrite(A_STBY, LOW);
  digitalWrite(B_STBY, LOW);

  pinMode(M1_IN1, OUTPUT);
  pinMode(M1_IN2, OUTPUT);
  bool pwm1Ready = ledcAttach(M1_PWM, 1000, 8);

  pinMode(M2_IN1, OUTPUT);
  pinMode(M2_IN2, OUTPUT);
  bool pwm2Ready = ledcAttach(M2_PWM, 1000, 8);

  pinMode(M3_IN1, OUTPUT);
  pinMode(M3_IN2, OUTPUT);
  bool pwm3Ready = ledcAttach(M3_PWM, 1000, 8);

  if (!pwm1Ready || !pwm2Ready || !pwm3Ready) {
    Serial.println("[Motor] PWM setup failed; drivers remain in standby.");
    while (true) {
      delay(1000);
    }
  }

  stopCar();
  digitalWrite(A_STBY, HIGH);
  digitalWrite(B_STBY, HIGH);
}

void applyMotionCommand(const String &command) {
  if (command == "F") {
    moveForward();
  } else if (command == "B") {
    moveBackward();
  } else if (command == "LF") {
    moveLeftForward();
  } else if (command == "RF") {
    moveRightForward();
  } else if (command == "LB") {
    moveLeftBackward();
  } else if (command == "RB") {
    moveRightBackward();
  } else {
    stopCar();
  }
}

void handleMessage(const String &message) {
  String command = message;
  command.trim();
  command.toUpperCase();

  if (command.length() == 0) {
    return;
  }

  if (command.startsWith("SPD ")) {
    int newSpeed = command.substring(4).toInt();
    if (newSpeed >= 0 && newSpeed <= 255) {
      speedVal = newSpeed;
      Serial.print("Speed set to ");
      Serial.println(speedVal);
    }
    lastCommandMs = millis();
    return;
  }

  if (command == "PING") {
    if (activeClient && activeClient.connected()) {
      activeClient.print("PONG id=");
      activeClient.println(CAR_ID);
    }
    return;
  }

  applyMotionCommand(command);
  lastCommandMs = millis();
  Serial.print("CMD: ");
  Serial.println(command);
}

void beginDiscovery() {
  discoveryUdp.begin(DISCOVERY_PORT);
  Serial.print("Discovery UDP port: ");
  Serial.println(DISCOVERY_PORT);
}

void sendDiscoveryReply(IPAddress remoteIp, uint16_t remotePort) {
  String reply = "BOID_CAR id=" + String(CAR_ID);
  reply += " name=";
  reply += CAR_NAME;
  reply += " ip=";
  reply += WiFi.localIP().toString();
  reply += " tcp=";
  reply += String(CONTROL_PORT);

  discoveryUdp.beginPacket(remoteIp, remotePort);
  discoveryUdp.print(reply);
  discoveryUdp.endPacket();
}

void handleDiscovery() {
  int packetSize = discoveryUdp.parsePacket();
  if (packetSize <= 0) {
    return;
  }

  char buffer[64];
  int len = discoveryUdp.read(buffer, sizeof(buffer) - 1);
  if (len <= 0) {
    return;
  }
  buffer[len] = '\0';

  String request = String(buffer);
  request.trim();
  if (request == DISCOVERY_REQUEST) {
    sendDiscoveryReply(discoveryUdp.remoteIP(), discoveryUdp.remotePort());
  }
}

void require5GHzOnly() {
  // Set the band after starting STA, and before every explicit connection.
  // Never fall back to 2.4GHz when YAYA's 5GHz radio is unavailable.
  if (!WiFi.setBandMode(WIFI_BAND_MODE_5G_ONLY) ||
      WiFi.getBandMode() != WIFI_BAND_MODE_5G_ONLY) {
    stopCar();
    digitalWrite(A_STBY, LOW);
    digitalWrite(B_STBY, LOW);
    WiFi.setAutoReconnect(false);
    WiFi.disconnect(true);
    Serial.println("[WiFi] Cannot enforce 5GHz-only; stopped. Check board/core settings.");
    while (true) {
      delay(1000);
    }
  }
}

void connectToWifi() {
  WiFi.onEvent(onWiFiEvent);
  WiFi.persistent(false);
  WiFi.setHostname(CAR_NAME);
  WiFi.mode(WIFI_STA);
  require5GHzOnly();
  WiFi.setAutoReconnect(true);
  WiFi.setSleep(false);
  Serial.println("[WiFi] 5GHz ONLY; 2.4GHz disabled.");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print(CAR_NAME);
  Serial.print(" connecting to ");
  Serial.println(WIFI_SSID);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    attempts++;
    if (attempts % 20 == 0) {
      Serial.print(" status=");
      Serial.print(WiFi.status());
      Serial.print(" rssi=");
      Serial.println(WiFi.RSSI());
    }
  }

  Serial.println();
  Serial.print(CAR_NAME);
  Serial.print(" IP: ");
  Serial.println(WiFi.localIP());
}

void setup() {
  Serial.begin(115200);

  initMotors();
  connectToWifi();
  beginDiscovery();
  server.begin();
  server.setNoDelay(true);
  commandBuffer.reserve(48);
  lastCommandMs = millis();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    stopCar();
    unsigned long now = millis();
    if (now - lastReconnectAttemptMs > 2000) {
      lastReconnectAttemptMs = now;
      Serial.println("WiFi lost, reconnecting...");
      WiFi.disconnect();
      require5GHzOnly();
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    }
    delay(50);
    return;
  }

  handleDiscovery();

  unsigned long now = millis();
  if (now - lastRssiLogMs >= 5000) {
    lastRssiLogMs = now;
    Serial.print("[WiFi] RSSI=");
    Serial.println(WiFi.RSSI());
  }

  if (!activeClient || !activeClient.connected()) {
    if (activeClient) {
      activeClient.stop();
      commandBuffer = "";
    }
    activeClient = server.available();
    if (activeClient) {
      activeClient.setNoDelay(true);
      lastClientRxMs = millis();
      Serial.print("[TCP] connected ");
      Serial.println(CAR_NAME);
    }
  }

  if (activeClient && activeClient.connected()) {
    while (activeClient.available()) {
      char incoming = static_cast<char>(activeClient.read());
      lastClientRxMs = millis();

      if (incoming == '\r' || incoming == '\n') {
        if (commandBuffer.length() > 0) {
          handleMessage(commandBuffer);
          commandBuffer = "";
        }
        continue;
      }

      if (incoming >= 32 && incoming <= 126) {
        commandBuffer += incoming;
      }
    }
  }

  if (activeClient && activeClient.connected() && millis() - lastClientRxMs > CLIENT_IDLE_TIMEOUT_MS) {
    Serial.println("[TCP] idle timeout, closing client");
    stopCar();
    activeClient.stop();
    commandBuffer = "";
  }

  if (millis() - lastCommandMs > COMMAND_TIMEOUT_MS) {
    stopCar();
  }

  delay(15);
}
