#include <Arduino.h>

#if !defined(CONFIG_IDF_TARGET_ESP32C5)
#error "Select an ESP32-C5 board. For old ESP32 cars use firmware/biedong/biedong.ino."
#endif

// ESP32-C5-WIFI6-KIT-N16R8-M offline motor test. No Wi-Fi or Vicon required.
// Wiring matches car16-car50 C5 firmware: see docs/ESP32_C5_WIRING.md.
// Separate STBY: driver A to GPIO23, driver B to GPIO25. Motors start automatically.
// Same speed, six directions and timing as the original biedong sketch.

// ===== TB6612-A：电机 M1、M2 =====
const int A_STBY = 23;

// M1
const int M1_IN1 = 0;
const int M1_IN2 = 1;
const int M1_PWM = 3;

// M2
const int M2_IN1 = 4;
const int M2_IN2 = 5;
const int M2_PWM = 8;

// ===== TB6612-B：电机 M3 =====
const int B_STBY = 25;

// M3
const int M3_IN1 = 9;
const int M3_IN2 = 10;
const int M3_PWM = 24;

const int PWM_FREQ = 1000;
const int PWM_RES  = 8;

int speedVal = 150;

// ===== 基本控制函数 =====
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

void stopAll() {
  motorStop(M1_IN1, M1_IN2, M1_PWM);
  motorStop(M2_IN1, M2_IN2, M2_PWM);
  motorStop(M3_IN1, M3_IN2, M3_PWM);
}

// ===== 运动函数 =====

// 前进
void moveForward() {
  motorStop(M1_IN1, M1_IN2, M1_PWM);
  motorBackward(M2_IN1, M2_IN2, M2_PWM);
  motorBackward(M3_IN1, M3_IN2, M3_PWM);
}

// 后退
void moveBackward() {
  motorStop(M1_IN1, M1_IN2, M1_PWM);
  motorForward(M2_IN1, M2_IN2, M2_PWM);
  motorForward(M3_IN1, M3_IN2, M3_PWM);
}

// 左上移
void moveLeftForward() {
  motorBackward(M1_IN1, M1_IN2, M1_PWM);
  motorStop(M2_IN1, M2_IN2, M2_PWM);
  motorBackward(M3_IN1, M3_IN2, M3_PWM);
}

// 右下移
void moveRightBackward() {
  motorForward(M1_IN1, M1_IN2, M1_PWM);
  motorStop(M2_IN1, M2_IN2, M2_PWM);
  motorForward(M3_IN1, M3_IN2, M3_PWM);
}

// 左下移
void moveLeftBackward() {
  motorBackward(M1_IN1, M1_IN2, M1_PWM);
  motorForward(M2_IN1, M2_IN2, M2_PWM);
  motorStop(M3_IN1, M3_IN2, M3_PWM);
}

// 右上移
void moveRightForward() {
  motorForward(M1_IN1, M1_IN2, M1_PWM);
  motorBackward(M2_IN1, M2_IN2, M2_PWM);
  motorStop(M3_IN1, M3_IN2, M3_PWM);
}
void setup() {
  Serial.begin(115200);
  pinMode(A_STBY, OUTPUT);
  pinMode(B_STBY, OUTPUT);

  digitalWrite(A_STBY, LOW);
  digitalWrite(B_STBY, LOW);

  pinMode(M1_IN1, OUTPUT);
  pinMode(M1_IN2, OUTPUT);
  bool pwm1Ready = ledcAttach(M1_PWM, PWM_FREQ, PWM_RES);

  pinMode(M2_IN1, OUTPUT);
  pinMode(M2_IN2, OUTPUT);
  bool pwm2Ready = ledcAttach(M2_PWM, PWM_FREQ, PWM_RES);

  pinMode(M3_IN1, OUTPUT);
  pinMode(M3_IN2, OUTPUT);
  bool pwm3Ready = ledcAttach(M3_PWM, PWM_FREQ, PWM_RES);

  if (!pwm1Ready || !pwm2Ready || !pwm3Ready) {
    Serial.println("[Motor] PWM setup failed; drivers remain in standby.");
    while (true) {
      delay(1000);
    }
  }

  stopAll();
  digitalWrite(A_STBY, HIGH);
  digitalWrite(B_STBY, HIGH);
  Serial.println("C5 motor test: F, B, LF, RB, LB, RF; repeats automatically. PWM=150.");
}

void loop() {

  // 前
  moveForward();
  delay(2000);
  stopAll();
  delay(1000);

  // 后
  moveBackward();
  delay(2000);
  stopAll();
  delay(1000);

  // 左上
  moveLeftForward();
  delay(2000);
  stopAll();
  delay(1000);

  // 右下
  moveRightBackward();
  delay(2000);
  stopAll();
  delay(2000);
 
  // 左下
  moveLeftBackward();
  delay(2000);
  stopAll();
  delay(2000);
   
  // 右上
  moveRightForward();
  delay(2000);
  stopAll();
  delay(2000);
}