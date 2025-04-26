# /home/aleeya/FYP/fyp2/FYP/components/Kinectv2_motors.py
import time
import serial # <<< Make sure this line is uncommented >>>
import atexit # <<< Add this if you want automatic cleanup >>>

class motors:
    def __init__(self, serial_port='/dev/ttyACM0', baud_rate=9600):
        """
        Initializes the motor control interface.
        Replace this with your actual motor hardware setup.
        """
        self.motor_enabled = False
        self.arduino = None # <<< Initialize arduino attribute >>>
        self.port = serial_port # <<< Store port for cleanup >>>
        print("Initializing Motors...")

        # --- Example: Serial Connection (Uncomment and adapt if needed) ---
        # vvv UNCOMMENT THIS BLOCK vvv
        try:
            print(f"Attempting to connect to Arduino on {serial_port} at {baud_rate} baud...") # <<< Add print >>>
            self.arduino = serial.Serial(serial_port, baud_rate, timeout=1)
            time.sleep(2) # Wait for connection to establish/Arduino reset
            print(f"Successfully connected to motors via {serial_port}")
            self.motor_enabled = True
            atexit.register(self.cleanup) # <<< Register cleanup >>>
        except serial.SerialException as e:
            print(f"ERROR: Failed to connect to motors on {serial_port}: {e}")
            print("Motor commands will be simulated.")
            self.motor_enabled = False

    def _send_command(self, command):
        """ Helper function to send commands (adapt for your hardware) """
        if self.motor_enabled and self.arduino and self.arduino.is_open: # <<< Check arduino object too >>>
            # --- Example: Serial Command ---
            # vvv UNCOMMENT THIS BLOCK vvv
            try:
                full_command = f"{command}\n"
                self.arduino.write(full_command.encode('utf-8')) # <<< Specify encoding >>>
                # print(f"Sent: {command}") # Optional: print sent command
                # time.sleep(0.05) # Optional small delay
            except Exception as e:
                print(f"Error sending command '{command}': {e}")
            # ^^^ UNCOMMENT THIS BLOCK ^^^
            # -------------------------------

            # --- Simulation (Remove or keep commented if using hardware) ---
            # print(f"Simulating Command: {command}")
            # pass

        else:
            # Print a warning if trying to send command without connection
            if not self.motor_enabled:
                print(f"Warning: Motor hardware connection failed. Simulating command: {command}")
            else: # Should not happen if init is correct
                 print(f"Warning: Arduino not available. Cannot send command: {command}")

    def set_motor_speeds(self, left_speed, right_speed):
        """
        Sets the speed of the left and right motors individually.
        :param left_speed: Speed for the left motor (0-255).
        :param right_speed: Speed for the right motor (0-255).
        """
        print(f"Action: Set Motor Speeds (Left: {left_speed}, Right: {right_speed})")
        self._send_command(f"LEFT_SPEED:{int(left_speed)}")
        self._send_command(f"RIGHT_SPEED:{int(right_speed)}")

    def stop_motors(self):
        """Stops both motors."""
        print("Action: Stop Motors")
        self.set_motor_speeds(0, 0)

    def cleanup(self):
        """ Clean up resources (e.g., close serial port) """
        print("Cleaning up motor interface...")
        # --- Example: Close Serial ---
        # vvv UNCOMMENT THIS BLOCK vvv
        if self.motor_enabled and hasattr(self, 'arduino') and self.arduino and self.arduino.is_open:
            try:
                # Send a final stop command before closing
                print("Sending final stop command before closing port...")
                self.stop_motors()
                time.sleep(0.1) # Give time for command to be sent/processed
                print(f"Closing serial port {self.port}...")
                self.arduino.close()
                print("Serial port closed.")
            except Exception as e:
                print(f"Error closing serial port: {e}")
        # ^^^ UNCOMMENT THIS BLOCK ^^^
        # -----------------------------
        elif hasattr(self, 'arduino') and self.arduino:
             print("Serial port was not open.")
        else:
             print("No serial connection was established.")


# --- Remove or comment out the old Servo Example ---
