# /home/aleeya/FYP/fyp2/FYP/components/Kinectv2_motors.py
import time
# import serial # Uncomment if using serial communication (like for Arduino)

class motors:
    def __init__(self, serial_port='/dev/ttyACM0', baud_rate=9600):
        """
        Initializes the motor control interface.
        Replace this with your actual motor hardware setup.
        """
        self.motor_enabled = False
        print("Initializing Motors...")

        # --- Example: Serial Connection (Uncomment and adapt if needed) ---
        # try:
        #     self.arduino = serial.Serial(serial_port, baud_rate, timeout=1)
        #     time.sleep(2) # Wait for connection to establish
        #     print(f"Successfully connected to motors via {serial_port}")
        #     self.motor_enabled = True
        # except serial.SerialException as e:
        #     print(f"ERROR: Failed to connect to motors on {serial_port}: {e}")
        #     print("Motor commands will be simulated.")
        #     self.motor_enabled = False
        # --------------------------------------------------------------------

        # If no hardware connection, motor commands will just print
        if not self.motor_enabled:
             print("Motor hardware not connected. Commands will be simulated.")
             self.motor_enabled = True # Set to True to allow simulation prints

        print("Motors Initialized (Simulated or Connected).")


    def _send_command(self, command):
        """ Helper function to send commands (adapt for your hardware) """
        if self.motor_enabled:
            # --- Example: Serial Command ---
            # try:
            #     full_command = f"{command}\n"
            #     self.arduino.write(full_command.encode())
            #     # print(f"Sent: {command}") # Optional: print sent command
            #     # time.sleep(0.05) # Optional small delay
            # except Exception as e:
            #     print(f"Error sending command '{command}': {e}")
            # -------------------------------

            # --- Simulation ---
            # print(f"Simulating Command: {command}") # Keep this for simulation
            pass # In simulation, just printing the action is enough (done in methods below)

        else:
            # This case should ideally not be reached if constructor handles init failure
            print("Motor interface not enabled. Cannot send command.")

    def move_forward(self):
        """ Commands the robot to move forward. """
        print("Action: Move Forward") # Keep this print
        # Add your actual hardware command here
        self._send_command("FORWARD") # Example command string

    def turn_left(self):
        """ Commands the robot to turn left. """
        print("Action: Turn Left") # Keep this print
        # Add your actual hardware command here
        self._send_command("LEFT") # Example command string

    def turn_right(self):
        """ Commands the robot to turn right. """
        print("Action: Turn Right") # Keep this print
        # Add your actual hardware command here
        self._send_command("RIGHT") # Example command string

    def stop_motors(self):
        """ Commands the robot to stop all movement. """
        print("Action: Stop Motors") # Keep this print
        # Add your actual hardware command here
        self._send_command("STOP") # Example command string

    def cleanup(self):
        """ Clean up resources (e.g., close serial port) """
        print("Cleaning up motor interface...")
        # --- Example: Close Serial ---
        # if self.motor_enabled and hasattr(self, 'arduino') and self.arduino.is_open:
        #     try:
        #         self.arduino.close()
        #         print("Serial port closed.")
        #     except Exception as e:
        #         print(f"Error closing serial port: {e}")
        # -----------------------------


# --- Original Servo Example (for reference if needed) ---
# import serial
# import time
# # Connect to Arduino's serial port
# arduino = serial.Serial('/dev/ttyACM0', 9600)
# time.sleep(2)  # Wait for Arduino to reset
# def set_servo_angle(angle):
#     command = f"servo:{angle}\n"
#     arduino.write(command.encode())
#     print(f"Sent: {command.strip()}")
# # Example usage
# set_servo_angle(90)
# time.sleep(1)
# set_servo_angle(0)
# arduino.close()
# --- End Servo Example ---

