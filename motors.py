import serial
import time

# Connect to Arduino's serial port
arduino = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)  # Wait for Arduino to reset

def set_servo_angle(angle):
    command = f"servo:{angle}\n"
    arduino.write(command.encode())
    print(f"Sent: {command.strip()}")

# Example usage
set_servo_angle(90)
time.sleep(1)
set_servo_angle(0)

arduino.close()
