#include <Arduino.h>

// --- Pin Definitions (MUST MATCH YOUR WIRING) ---
// Adjust these pins based on your Arduino board and IBT-2 connections
// Ensure these are PWM-capable pins (usually marked with '~')

// --- Motor A (e.g., Left Motor) Pins ---
const int MOTOR_A_RPWM_PIN = 5;  // Connect Arduino Pin 5 to IBT-2 (A) RPWM (Forward PWM)
const int MOTOR_A_LPWM_PIN = 6;  // Connect Arduino Pin 6 to IBT-2 (A) LPWM (Reverse PWM)

// --- Motor B (e.g., Right Motor) Pins ---
const int MOTOR_B_RPWM_PIN = 9;  // Connect Arduino Pin 9 to IBT-2 (B) RPWM (Forward PWM)
const int MOTOR_B_LPWM_PIN = 10; // Connect Arduino Pin 10 to IBT-2 (B) LPWM (Reverse PWM)

// --- Constants for Speed ---
const int MAX_PWM_SPEED = 255;
const int NORMAL_SPEED = 200; // Adjust as needed (0-255)
const int SLOW_SPEED = 100;   // Adjust as needed (0-255)
const int TURN_SPEED = 180;   // Speed for turning (adjust as needed)

// --- Function Prototypes ---
void setMotorASpeed(int speed); // Controls Motor A (-255 to 255)
void setMotorBSpeed(int speed); // Controls Motor B (-255 to 255)
void stopAllMotors();

// --- Setup Function ---
void setup() {
  Serial.begin(9600); // Must match the baud rate in motors.py
  while (!Serial); // Wait for serial connection

  // Configure motor control pins as OUTPUTs
  pinMode(MOTOR_A_RPWM_PIN, OUTPUT);
  pinMode(MOTOR_A_LPWM_PIN, OUTPUT);
  pinMode(MOTOR_B_RPWM_PIN, OUTPUT);
  pinMode(MOTOR_B_LPWM_PIN, OUTPUT);

  // Ensure motors are stopped initially
  stopAllMotors();
  Serial.println("Arduino Ready. Waiting for commands...");
}

// --- Main Loop ---
void loop() {
  // Check if data is available to read from serial
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.startsWith("LEFT_SPEED:")) {
      int left_speed = command.substring(11).toInt();
      setMotorASpeed(left_speed);
    } else if (command.startsWith("RIGHT_SPEED:")) {
      int right_speed = command.substring(12).toInt();
      setMotorBSpeed(right_speed);
    } else if (command == "STOP") {
      stopAllMotors();
    } else {
      Serial.println("Unknown command received.");
    }
  }
}

// --- Motor A Control Function (IBT-2 Specific) ---
void setMotorASpeed(int speed) {
  speed = constrain(speed, -MAX_PWM_SPEED, MAX_PWM_SPEED);
  if (speed > 0) { // Forward
    analogWrite(MOTOR_A_RPWM_PIN, speed);
    analogWrite(MOTOR_A_LPWM_PIN, 0);
  } else if (speed < 0) { // Reverse
    analogWrite(MOTOR_A_RPWM_PIN, 0);
    analogWrite(MOTOR_A_LPWM_PIN, -speed); // Use absolute value for PWM
  } else { // Stop (Coast)
    analogWrite(MOTOR_A_RPWM_PIN, 0);
    analogWrite(MOTOR_A_LPWM_PIN, 0);
  }
}

// --- Motor B Control Function (IBT-2 Specific) ---
void setMotorBSpeed(int speed) {
  speed = constrain(speed, -MAX_PWM_SPEED, MAX_PWM_SPEED);
  if (speed > 0) { // Forward
    analogWrite(MOTOR_B_RPWM_PIN, speed);
    analogWrite(MOTOR_B_LPWM_PIN, 0);
  } else if (speed < 0) { // Reverse
    analogWrite(MOTOR_B_RPWM_PIN, 0);
    analogWrite(MOTOR_B_LPWM_PIN, -speed); // Use absolute value for PWM
  } else { // Stop (Coast)
    analogWrite(MOTOR_B_RPWM_PIN, 0);
    analogWrite(MOTOR_B_LPWM_PIN, 0);
  }
}

// --- Stop Both Motors ---
void stopAllMotors() {
  setMotorASpeed(0);
  setMotorBSpeed(0);
}
