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
    // Read the incoming command string until newline character
    String command = Serial.readStringUntil('\n');
    command.trim(); // Remove leading/trailing whitespace and newline characters

    // Process the received command if it's not empty
    if (command.length() > 0) {
        Serial.print("Received Command: ");
        Serial.println(command); // Print the actual received command

        // --- Updated Command Checks ---
        if (command == "FORWARD") { // Check for "FORWARD" (used by move_forward and move_normal_forward)
          Serial.println("Action: Moving Forward (Normal)");
          setMotorASpeed(NORMAL_SPEED);
          setMotorBSpeed(NORMAL_SPEED);
        } else if (command == "FORWARD_SLOW") { // Check for "FORWARD_SLOW"
          Serial.println("Action: Moving Forward (Slow)");
          setMotorASpeed(SLOW_SPEED);
          setMotorBSpeed(SLOW_SPEED);
        } else if (command == "LEFT") { // Check for "LEFT"
          Serial.println("Action: Turning Left (Spin)");
          setMotorASpeed(-TURN_SPEED); // Spin turn: Left backward
          setMotorBSpeed(TURN_SPEED);  // Spin turn: Right forward
        } else if (command == "RIGHT") { // Check for "RIGHT"
          Serial.println("Action: Turning Right (Spin)");
          setMotorASpeed(TURN_SPEED);   // Spin turn: Left forward
          setMotorBSpeed(-TURN_SPEED); // Spin turn: Right backward
        } else if (command == "STOP") { // Check for "STOP"
          Serial.println("Action: Stopping Motors");
          stopAllMotors();
        }
        // Add other commands if needed (e.g., "BACKWARD")
        // else if (command == "BACKWARD") { ... }
         else { // Handle unknown commands
          Serial.println("Action: Unknown command received.");
          // Optional: Stop motors on unknown command for safety
          // stopAllMotors();
        }
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
