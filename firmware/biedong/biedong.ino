// ===== TB6612-A：电机 M1、M2 =====
const int A_STBY = 14;

// M1
const int M1_IN1 = 27;
const int M1_IN2 = 26;
const int M1_PWM = 25;

// M2
const int M2_IN1 = 33;
const int M2_IN2 = 32;
const int M2_PWM = 23;

// ===== TB6612-B：电机 M3 =====
const int B_STBY = 13;

// M3
const int M3_IN1 = 18;
const int M3_IN2 = 19;
const int M3_PWM = 5;

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
  pinMode(A_STBY, OUTPUT);
  pinMode(B_STBY, OUTPUT);

  digitalWrite(A_STBY, HIGH);
  digitalWrite(B_STBY, HIGH);

  pinMode(M1_IN1, OUTPUT);
  pinMode(M1_IN2, OUTPUT);
  ledcAttach(M1_PWM, PWM_FREQ, PWM_RES);

  pinMode(M2_IN1, OUTPUT);
  pinMode(M2_IN2, OUTPUT);
  ledcAttach(M2_PWM, PWM_FREQ, PWM_RES);

  pinMode(M3_IN1, OUTPUT);
  pinMode(M3_IN2, OUTPUT);
  ledcAttach(M3_PWM, PWM_FREQ, PWM_RES);

  stopAll();
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