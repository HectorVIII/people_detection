########################################################################
#
# Multi-person body tracking with OpenCV 2D viewer + manual multi-role
# binding using mouse clicks, with an in-window "Binding Help" button
# that opens a separate English tutorial dialog.
#
# Roles:
#   Surgeon_1, Surgeon_2, ...
#   Assistant_1, Assistant_2, ...
#   Nurse_1, Nurse_2, ...
#
# The input can be a live camera, an SVO file, or a stream from an IP address.
#
########################################################################

import cv2
import pyzed.sl as sl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import argparse


def parse_args(init, opt):
    """
    Parse command line options and configure ZED InitParameters.
    """
    # SVO file input
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input:", opt.input_svo_file)

    # Streaming input
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        # Basic check for IP:port or IP
        if (
            ip_str.replace(":", "").replace(".", "").isdigit()
            and len(ip_str.split(".")) == 4
            and len(ip_str.split(":")) == 2
        ):
            init.set_from_stream(ip_str.split(":")[0], int(ip_str.split(":")[1]))
            print("[Sample] Using Stream input, IP:", ip_str)
        elif ip_str.replace(":", "").replace(".", "").isdigit() and len(ip_str.split(".")) == 4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP:", ip_str)
        else:
            print("[Sample] Invalid IP format. Using live camera")

    # Resolution selection
    if "HD2K" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif "HD1200" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif "HD1080" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif "HD720" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif "SVGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif "VGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution) > 0:
        print("[Sample] No valid resolution entered. Using default")
    else:
        print("Using default resolution")


def create_binding_help_image(width=700, height=600):
    """
    Create a black image with white English instructions
    explaining how to bind people to roles.
    """
    help_img = np.zeros((height, width, 3), dtype=np.uint8) # Black background

    lines = [
        "Body Role Binding Tutorial",
        "----------------------------------------",
        "1) All people stand in front of the camera.",
        "2) Press 's' to select the Surgeon role,",
        "   then LEFT-CLICK on a person to bind:",
        "     -> Surgeon_1, Surgeon_2, etc.",
        "",
        "3) Press 'a' for Assistant, 'n' for Nurse,",
        "   then LEFT-CLICK on each person to bind:",
        "     -> Assistant_1, Nurse_1, etc.",
        "",
        "4) You can press ESC to cancel the current role selection.",
        "",
        "5) The label will be shown above the skeleton.",
        "",
        "6) You can re-bind a person by selecting a role",
        "   and clicking on them again.",
        "",
        "7) Press 'q' in the main window to quit the app.",
        "",
        "IMPORTANT:",
        "  If a person leaves the camera view completely",
        "  and then comes back, their tracking ID may change,",
        "  so you may need to bind that person again."
    ]   # Instruction lines

    y0 = 30 # Initial y position
    dy = 24 # Line spacing

    for i, line in enumerate(lines):
        y = y0 + i * dy # Calculate y position for this line
        cv2.putText(    
            help_img,   
            line,   
            (20, y),    
            cv2.FONT_HERSHEY_SIMPLEX,   
            0.6,    
            (255, 255, 255),    
            1,  
            cv2.LINE_AA,    
        )   # Draw the text line on the image

    return help_img 


def main(opt):  
    print("  - Press 'q' in the main window to quit.")
    print("  - Press 'm' to pause/resume.")

    # Create a Camera object
    zed = sl.Camera()

    # Create InitParameters and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL        # Use NEURAL depth mode
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # Use Y-up coordinate system

    # Parse CLI arguments into init_params
    parse_args(init_params, opt)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera open error:", err)
        exit(1)

    # Enable Positional tracking (mandatory for body tracking)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, you can improve performance with:
    positional_tracking_parameters.set_as_static = True

    zed.enable_positional_tracking(positional_tracking_parameters)

    # Configure body tracking parameters
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                 # Track people across frames
    body_param.enable_body_fitting = True            # enables the fitting process of each detected person
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE  # state-of-the-art accuracy, requires powerful GPU
    """
    BODY_TRACKING_MODEL::HUMAN_BODY_FAST: real time performance even on NVIDIA® Jetson™ or low-end GPU cards

    BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM: this is a compromise between accuracy and speed

    BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE: state-of-the-art accuracy, requires powerful GPU
    """
    body_param.body_format = sl.BODY_FORMAT.BODY_38   # Skeleton format (38 keypoints)

    # Enable body tracking module
    zed.enable_body_tracking(body_param)

    # Runtime body tracking parameters
    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 70  # Minimum confidence to detect a body

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities: downscale image for display (for performance)
    """
    display_resolution = sl.Resolution(
        min(camera_info.camera_configuration.resolution.width, 1280),
        min(camera_info.camera_configuration.resolution.height, 720)
    )   # Display resolution (max 1280x720)
    """
    display_resolution = camera_info.camera_configuration.resolution    # Use full resolution for display
    image_scale = [
        display_resolution.width / camera_info.camera_configuration.resolution.width,
        display_resolution.height / camera_info.camera_configuration.resolution.height,
    ]   # Scaling factors for width and height
       

    # ZED objects
    bodies = sl.Bodies()    # To store detected bodies
    image = sl.Mat()    # To store the left image
    key_wait = 10  # delay for cv2.waitKey in milliseconds

    # ------------------------------------------------------------------
    # Manual role binding state:
    #   role_map: ZED body ID -> role string ("Surgeon_1", "Assistant_2", ...)
    #   current_role: base role string for the next click ("Surgeon", "Assistant", "Nurse")
    #   click_pending: True if user has clicked and we need to assign a role
    #   click_pos: (x, y) in display image coordinates
    #   surgeon_count / assistant_count / nurse_count: counters for numbering
    # ------------------------------------------------------------------
    role_map = {}   # body ID (int) -> role string
    current_role = None
    click_pending = False 
    click_pos = (0, 0)  # (x, y) of the last click

    role_color = {
        "Surgeon":  (0, 0, 255),   # RED
        "Assistant": (255, 0, 0),  # BLUE
        "Nurse":  (0, 255, 0)      # GREEN
    }

    surgeon_count = 0
    assistant_count = 0
    nurse_count = 0

    # ------------------------------------------------------------------
    # "Binding Help" button settings (drawn inside the main image)
    # ------------------------------------------------------------------
    button_x = 20
    button_y = 70
    button_w = 170
    button_h = 40

    binding_help_requested = False  # Flag to show help window

    window_name = "ZED | 2D View"
    help_window_name = "Binding Help"

    cv2.namedWindow(window_name)    # Create main OpenCV window

    # Mouse callback for selecting a body by clicking near it OR clicking the help button
    def mouse_callback(event, x, y, flags, param):
        # Handle left button click events
        nonlocal click_pending, click_pos, binding_help_requested   # Use nonlocal to modify outer variables
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the click is inside the Binding Help button
            if button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h:
                binding_help_requested = True
                print("[UI] Binding Help button clicked.")
            else:
                # Otherwise, treat this as a click for body binding
                click_pos = (x, y)
                click_pending = True
                print(f"[Mouse] Click at ({x}, {y})")

    cv2.setMouseCallback(window_name, mouse_callback)   # Set mouse callback for main window

    # Pre-create the help image so we can show it any time
    binding_help_img = create_binding_help_image()

    while True:
        # Grab an image
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve bodies
            zed.retrieve_bodies(bodies, body_runtime_param)

            # Count people in this frame
            num_people = len(bodies.body_list)

            # Get image for OpenCV
            image_left_ocv = image.get_data()

            # Draw skeletons in 2D using the helper
            cv_viewer.render_2D(
                image_left_ocv,
                image_scale,
                bodies.body_list,
                body_param.enable_tracking,
                body_param.body_format
            )

            # Draw total number of people (top-left corner)
            cv2.putText(
                image_left_ocv,
                f"People detected: {num_people}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Draw the "Binding Help" button in the image
            # Button background (filled white)
            cv2.rectangle(
                image_left_ocv,
                (button_x, button_y),
                (button_x + button_w, button_y + button_h),
                (255, 255, 255),
                thickness=-1
            )
            # Button border (black)
            cv2.rectangle(
                image_left_ocv,
                (button_x, button_y),
                (button_x + button_w, button_y + button_h),
                (0, 0, 0),
                thickness=2
            )
            # Button text
            cv2.putText(
                image_left_ocv,
                "Binding Help",
                (button_x + 15, button_y + 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

            # Show current selected role at the bottom-left corner
            if current_role is not None:
                cv2.putText(
                    image_left_ocv,
                    f"Current role to bind: {current_role}",
                    (20, display_resolution.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # Draw ID / role label for each body
            for body in bodies.body_list:
                body_id = body.id

                # Get a reference 2D keypoint (e.g. root joint) as anchor for the label
                if body.keypoint_2d is not None and len(body.keypoint_2d) > 0:
                    kp = body.keypoint_2d[0]  # [x, y] in original resolution
                    if not np.isnan(kp[0]) and not np.isnan(kp[1]):
                        x = int(kp[0] * image_scale[0])
                        y = int(kp[1] * image_scale[1])

                        # Use assigned role if available, otherwise show raw ID
                        role_text = role_map.get(body_id, f"ID:{body_id}")

                        # Determine color based on role
                        if "Surgeon" in role_text:
                            color = role_color["Surgeon"]
                        elif "Assistant" in role_text:
                            color = role_color["Assistant"]
                        elif "Nurse" in role_text:
                            color = role_color["Nurse"]
                        else:
                            color = (0, 255, 255)   # yellow for raw ID (no role)

                        cv2.putText(
                            image_left_ocv,
                            role_text,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                            cv2.LINE_AA
                        )

            # ------------------------------------------------------------------
            # If user clicked inside the Binding Help button, show the help window
            # ------------------------------------------------------------------
            if binding_help_requested:
                cv2.imshow(help_window_name, binding_help_img)
                binding_help_requested = False

            # ------------------------------------------------------------------
            # If user clicked somewhere else and a current_role is selected,
            # find the nearest body to the click and assign a numbered role.
            # ------------------------------------------------------------------
            if click_pending and current_role is not None and len(bodies.body_list) > 0:
                cx, cy = click_pos
                best_body = None
                best_dist2 = None

                # Find the nearest body based on the first keypoint distance to the click
                for body in bodies.body_list:
                    if body.keypoint_2d is None or len(body.keypoint_2d) == 0:
                        continue
                    kp = body.keypoint_2d[0]
                    if np.isnan(kp[0]) or np.isnan(kp[1]):
                        continue
                    bx = int(kp[0] * image_scale[0])
                    by = int(kp[1] * image_scale[1])
                    dx = bx - cx
                    dy = by - cy
                    dist2 = dx * dx + dy * dy
                    if best_dist2 is None or dist2 < best_dist2:
                        best_dist2 = dist2
                        best_body = body

                if best_body is not None:
                    bid = best_body.id

                    # Generate unique role name based on base role and counters
                    if current_role == "Surgeon":
                        surgeon_count += 1
                        assigned_role = f"Surgeon_{surgeon_count}"
                    elif current_role == "Assistant":
                        assistant_count += 1
                        assigned_role = f"Assistant_{assistant_count}"
                    elif current_role == "Nurse":
                        nurse_count += 1
                        assigned_role = f"Nurse_{nurse_count}"
                    else:
                        # Fallback: should not happen
                        assigned_role = current_role

                    # Save to role map
                    role_map[bid] = assigned_role

                    print(f"[Bind] Assigned role '{assigned_role}' to body ID {bid}")
                else:
                    print("[Bind] No valid body found near click.")

                # Reset the click flag (one click = one assignment)
                click_pending = False
                # Optional: reset current_role to force user to press s/a/n again
                # current_role = None

            # Show the main image
            cv2.imshow(window_name, image_left_ocv)

            # Handle keyboard
            key = cv2.waitKey(key_wait) & 0xFF

            # Quit
            if key == ord('q'):
                print("Exiting...")
                break

            # Pause / resume
            if key == ord('m'):
                if key_wait > 0:
                    print("Pause")
                    key_wait = 0
                else:
                    print("Restart")
                    key_wait = 10

            # ----------------------------------------------------------
            # Role selection: press s/a/n to choose which role
            # will be assigned on the next mouse click.
            # ----------------------------------------------------------
            if key == ord('s'):
                current_role = "Surgeon"
                print("[Role] Current role set to 'Surgeon'. Click on a person to bind.")
            elif key == ord('a'):
                current_role = "Assistant"
                print("[Role] Current role set to 'Assistant'. Click on a person to bind.")
            elif key == ord('n'):
                current_role = "Nurse"
                print("[Role] Current role set to 'Nurse'. Click on a person to bind.")
            elif key == 27:  # ESC key to cancel current role selection
                current_role = None
                print("[Role] Binding cancelled.")

    # Clean up
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  #Create a command-line argument parser to define and handle user input parameters
    parser.add_argument(
        '--input_svo_file',
        type=str,
        help='Path to an .svo file, if you want to replay it',
        default=''
    )
    parser.add_argument(
        '--ip_address',
        type=str,
        help='IP Address, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup',
        default=''
    )
    parser.add_argument(
        '--resolution',
        type=str,
        help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA',
        default=''
    )
    opt = parser.parse_args()   #Parse the command-line arguments into the opt object

    # Safety check: do not specify both SVO and IP at the same time
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()

    main(opt)

