# Run "sudo pigpiod" before running script
import board
import cv2
import RPi.GPIO as GPIO
import time

from adafruit_motor import stepper
from adafruit_motorkit import MotorKit
from numpy import interp
from piservo import Servo


class MotionTracker:
    def __init__(self, x_starting_position: int = 200, y_starting_position: int = 200):
        """Init motion tracker.

        :param x_starting_position: X coordinate of starting point of motion tracker.
        :param y_starting_position: Y coordinate of starting point of motion tracker.
        """
        # Starting position of motors
        self.x_starting_position = x_starting_position
        self.y_starting_position = y_starting_position
        # Actual position of motors
        self.x_actual_position = x_starting_position
        self.y_actual_position = y_starting_position
        # Actual position of target
        self.x_target_position = 200
        self.y_target_position = 200
        # Information about if shot was taken and if servo is moving
        self.if_shoot = False
        self.if_servo_moving = False
        GPIO.setmode(GPIO.BCM)
        # Relay output
        GPIO.setup(4, GPIO.OUT)
        # Servo output
        self.x_servo_pwm = Servo(13)
        self.y_servo_pwm = Servo(12)

    def calibration(self):
        """Return tracker to init position."""
        calibration_diff = self.calculate_movement(self.x_starting_position, self.y_starting_position)

        self.motor_move(calibration_diff[0], 200, calibration_diff[1], True)
        self.motor_move(calibration_diff[2], 200, calibration_diff[3], True)

    def servo_move(self, x_position: int, y_position: int):
        """Move given servo to exact calculated position.

        :param x_position: X coordinate to which servo should move.
        :param y_position: Y coordinate to which servo should move.
        """
        # Max values for servo, should be adjusted to camera FOV so pixel(0,0) should be covered by both servos value 0
        x_servo_max_value = 110
        x_servo_min_value = 70
        y_servo_max_value = 110
        y_servo_min_value = 70

        # Map values from camera pixels(640x480) to values that should be sent to servo(servo_max_value, servo_min_value)
        x_calculation = int(interp(x_position, [0, 640], [x_servo_max_value, x_servo_min_value]))
        y_calculation = int(interp(y_position, [0, 480], [y_servo_max_value, y_servo_min_value]))

        # Write signal to servo
        self.x_servo_pwm.write(x_calculation)
        self.y_servo_pwm.write(y_calculation)

        # Change actual position of turret to which was just sent to servo
        self.x_actual_position = x_position
        self.y_actual_position = y_position

        self.if_shoot = True
        self.if_servo_moving = True

    def calculate_movement(self, x_position: int, y_position: int):
        """Calculate movement of motion tracker based on actual and target position.

        :param x_position: X coordinate of actual target.
        :param y_position: Y coordinate of actual target.

        :return tuple[str, int, str, int]: Directions and calculated movement for each motor.
        """
        x_calculation = self.x_actual_position - x_position
        y_calculation = self.y_actual_position - y_position

        x_movement = abs(x_calculation)
        y_movement = abs(y_calculation)

        if x_calculation > 0:
            x_direction = "LEFT"
        else:
            x_direction = "RIGHT"
        if y_calculation > 0:
            y_direction = "DOWN"
        else:
            y_direction = "UP"

        return x_direction, x_movement, y_direction, y_movement

    def motor_move(self, direction: str, speed: int, steps: int, calibration: bool):
        """Send signals to motor and make it move.

        :param direction: Direction of motor.
        :param speed: Designated speed of motor.
        :param steps: Number of steps to be send to motor.
        :param calibration: States if this is calibration(omits steps limit). True if yes, otherwise False.
        """
        if steps > 0:
            # Declare usage of stepper motor
            kit = MotorKit()
            # Calculate step timeout based on given speed
            step_timeout = 1/speed
            if steps > 5 and calibration is False:
                steps = 5
            self.if_shoot = False

            if direction == "LEFT":
                for i in range(steps):
                    kit.stepper1.onestep(direction=stepper.BACKWARD)
                    self.x_actual_position -= 1
                    time.sleep(step_timeout)
            elif direction == "RIGHT":
                for i in range(steps):
                    kit.stepper1.onestep(direction=stepper.FORWARD)
                    time.sleep(step_timeout)
                    self.x_actual_position += 1
            elif direction == "UP":
                for i in range(steps):
                    kit.stepper2.onestep(direction=stepper.FORWARD)
                    time.sleep(step_timeout)
                    self.y_actual_position += 1
            elif direction == "DOWN":
                for i in range(steps):
                    kit.stepper2.onestep(direction=stepper.BACKWARD)
                    time.sleep(step_timeout)
                    self.y_actual_position -= 1

    def draw_crosshair(self, frame: cv2.VideoCapture.read):
        """Draw crosshair for target and turret position.

        :param frame: Dataframe taken from video input.
        :return frame: Dataframe with added crosshairs.
        """
        # Draw crosshair for motion detection. Firstly - vertical, secondly - horizontal
        cv2.line(frame, (self.x_target_position - 10, self.y_target_position - 10),
                 (self.x_target_position + 10, self.y_target_position + 10),
                 (0, 0, 255), 3)
        cv2.line(frame, (self.x_target_position + 10, self.y_target_position - 10),
                 (self.x_target_position - 10, self.y_target_position + 10),
                 (0, 0, 255), 3)

        # Draw crosshair for turret position. Firstly - vertical, secondly - horizontal
        cv2.line(frame, (self.x_actual_position, self.y_actual_position - 10),
                 (self.x_actual_position, self.y_actual_position + 10),
                 (255, 0, 0), 3)
        cv2.line(frame, (self.x_actual_position - 10, self.y_actual_position),
                 (self.x_actual_position + 10, self.y_actual_position),
                 (255, 0, 0), 3)
        return frame

    def start_video_capture(self):
        """Start video capture and analysis."""
        cap = cv2.VideoCapture(0)
        mog = cv2.createBackgroundSubtractorMOG2()

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Add filters to detect motion on image
            fgmask = mog.apply(gray)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            fgmask = cv2.dilate(fgmask, kernel, iterations=1)

            contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate biggest contour on frame
            biggest_contour_area = 0
            for contour in contours:
                if cv2.contourArea(contour) > biggest_contour_area:
                    biggest_contour_area = cv2.contourArea(contour)
            self.if_shoot = False
            for contour in contours:
                # Ignore small contours
                if cv2.contourArea(contour) >= biggest_contour_area and cv2.contourArea(contour) > 1000:

                    # Draw bounding box around target
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Calculate center of target
                    self.x_target_position = int((2 * x + w) / 2)
                    self.y_target_position = int((2 * y + h) / 2)
                    self.servo_move(self.x_target_position, self.y_target_position)

                else:
                    continue

            # Calculate difference between target and actual position
            crosshair_diff = self.calculate_movement(self.x_target_position, self.y_target_position)

            # Calculate if position is +-target, if yes, then shoot
            if crosshair_diff[1] < 15 and crosshair_diff[3] < 15 and not self.if_shoot and self.if_servo_moving:
                GPIO.output(4, GPIO.HIGH)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'Shoot!', (100, 100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

                frame = self.draw_crosshair(frame)
                cv2.imshow('Motion Tracker', frame)
                self.if_servo_moving = False
                time.sleep(0.5)
                GPIO.output(4, GPIO.LOW)
            else:
                frame = self.draw_crosshair(frame)
                cv2.imshow('Motion Tracker', frame)
            # Move motors to reduce difference between target and actual position
            # self.motor_move(crosshair_diff[0], 200, crosshair_diff[1], False)
            # self.motor_move(crosshair_diff[2], 200, crosshair_diff[3], False)

            if cv2.waitKey(1) == ord('q'):
                GPIO.cleanup()
                break

        cap.release()
        cv2.destroyAllWindows()


# Init motion tracker with 640x480px camera
motion_tracker = MotionTracker(x_starting_position=320, y_starting_position=240)
motion_tracker.start_video_capture()
