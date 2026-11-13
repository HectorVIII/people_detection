#!/usr/bin/env python3
"""
All-in-one ZED body tracking + manual role binding demo.

This single script merges the logic from:
- args_parser.py
- binding_manager.py
- help_window.py
- viewer_2d.py
- zed_body_tracking.py
- main.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import pyzed.sl as sl
import cv_viewer.tracking_viewer as cv_viewer


# ==========================================================
# Argument parser
# ==========================================================
def configure_init_params(init: sl.InitParameters, opt: argparse.Namespace) -> None:
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

        if (
            ip_str.replace(":", "").replace(".", "").isdigit()
            and len(ip_str.split(".")) == 4
            and len(ip_str.split(":")) == 2
        ):
            init.set_from_stream(ip_str.split(":")[0], int(ip_str.split(":")[1]))
            print("[Sample] Using Stream input, IP:", ip_str)

        elif (
            ip_str.replace(":", "").replace(".", "").isdigit()
            and len(ip_str.split(".")) == 4
        ):
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP:", ip_str)

        else:
            print("[Sample] Invalid IP format. Using live camera")

    # Wired camera input
    elif opt.camera_serial_number > 0:
        init.set_from_serial_number(opt.camera_serial_number)
        print("[Sample] Using Camera serial number:", opt.camera_serial_number)

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
        print("[Sample] Using default resolution")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_svo_file",
        type=str,
        help="Path to an .svo file, if you want to replay it",
        default=""
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        help="IP Address in format a.b.c.d:port or a.b.c.d",
        default=""
    )
    parser.add_argument(
        "--resolution",
        type=str,
        help="Resolution: HD2K, HD1200, HD1080, HD720, SVGA or VGA",
        default=""
    )
    parser.add_argument(
        "--camera_serial_number",
        type=int,
        help="ZED camera serial number for wired camera input",
        default=32483963
    )
    return parser


# ==========================================================
# Help window
# ==========================================================
def create_binding_help_image(width: int = 700, height: int = 600) -> np.ndarray:
    """
    Create a black image with white English instructions
    explaining how to bind people to roles.
    """
    help_img = np.zeros((height, width, 3), dtype=np.uint8)

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
        "4) Press ESC to cancel the current role selection.",
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
    ]

    y0 = 30
    dy = 24

    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(
            help_img,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return help_img


# ==========================================================
# Role binding manager
# ==========================================================
@dataclass
class RoleBindingManager:
    """
    Manage manual role binding between ZED body IDs and semantic roles
    (Surgeon_i, Assistant_j, Nurse_k).
    """
    image_scale: Tuple[float, float]
    role_map: Dict[int, str] = field(default_factory=dict)
    current_role: Optional[str] = None
    click_pending: bool = False
    click_pos: Tuple[int, int] = (0, 0)

    surgeon_count: int = 0
    assistant_count: int = 0
    nurse_count: int = 0

    role_color: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "Surgeon": (0, 0, 255),    # Red
        "Assistant": (255, 0, 0),  # Blue
        "Nurse": (0, 255, 0)       # Green
    })

    def set_current_role_from_key(self, key: int) -> Optional[str]:
        """
        Map keyboard key to a role.
        """
        if key == ord("s"):
            self.current_role = "Surgeon"
            return "[Role] Current role set to 'Surgeon'. Click on a person to bind."

        if key == ord("a"):
            self.current_role = "Assistant"
            return "[Role] Current role set to 'Assistant'. Click on a person to bind."

        if key == ord("n"):
            self.current_role = "Nurse"
            return "[Role] Current role set to 'Nurse'. Click on a person to bind."

        if key == 27:  # ESC
            self.current_role = None
            return "[Role] Binding cancelled."

        return None

    def register_click(self, x: int, y: int) -> None:
        """
        Called by the mouse callback when the user clicks in the viewer.
        """
        self.click_pos = (x, y)
        self.click_pending = True
        print(f"[Mouse] Click at ({x}, {y})")

    def _generate_role_name(self) -> str:
        """
        Create a unique numbered role name based on current_role.
        """
        if self.current_role == "Surgeon":
            self.surgeon_count += 1
            return f"Surgeon_{self.surgeon_count}"

        if self.current_role == "Assistant":
            self.assistant_count += 1
            return f"Assistant_{self.assistant_count}"

        if self.current_role == "Nurse":
            self.nurse_count += 1
            return f"Nurse_{self.nurse_count}"

        return self.current_role or "Unknown"

    def try_assign_from_click(self, bodies: sl.Bodies) -> None:
        """
        If a click is pending and a role is selected, find the nearest body
        to the click position and assign the role.
        """
        if not self.click_pending or self.current_role is None:
            return

        if bodies is None or len(bodies.body_list) == 0:
            self.click_pending = False
            print("[Bind] No bodies available for assignment.")
            return

        cx, cy = self.click_pos
        best_body = None
        best_dist2 = None

        for body in bodies.body_list:
            if body.keypoint_2d is None or len(body.keypoint_2d) == 0:
                continue

            kp = body.keypoint_2d[0]
            if np.isnan(kp[0]) or np.isnan(kp[1]):
                continue

            bx = int(kp[0] * self.image_scale[0])
            by = int(kp[1] * self.image_scale[1])

            dx = bx - cx
            dy = by - cy
            dist2 = dx * dx + dy * dy

            if best_dist2 is None or dist2 < best_dist2:
                best_dist2 = dist2
                best_body = body

        if best_body is not None:
            bid = best_body.id
            assigned_role = self._generate_role_name()
            self.role_map[bid] = assigned_role
            print(f"[Bind] Assigned role '{assigned_role}' to body ID {bid}")
        else:
            print("[Bind] No valid body found near click.")

        self.click_pending = False

    def get_label_and_color(self, body_id: int) -> Tuple[str, Tuple[int, int, int]]:
        """
        Returns the text label and BGR color for a given body ID.
        """
        role_text = self.role_map.get(body_id, f"ID:{body_id}")

        if "Surgeon" in role_text:
            color = self.role_color["Surgeon"]
        elif "Assistant" in role_text:
            color = self.role_color["Assistant"]
        elif "Nurse" in role_text:
            color = self.role_color["Nurse"]
        else:
            color = (0, 255, 255)  # Yellow

        return role_text, color


# ==========================================================
# ZED body tracker
# ==========================================================
class ZEDBodyTracker:
    """
    Thin wrapper around ZED SDK body tracking configuration and frame retrieval.
    """

    def __init__(self, init_params: sl.InitParameters) -> None:
        self.zed = sl.Camera()
        self.init_params = init_params

        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Camera open error: {err}")

        positional_tracking_parameters = sl.PositionalTrackingParameters()
        positional_tracking_parameters.set_as_static = True
        self.zed.enable_positional_tracking(positional_tracking_parameters)

        self.body_param = sl.BodyTrackingParameters()
        self.body_param.enable_tracking = True
        self.body_param.enable_body_fitting = True
        self.body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        self.body_param.body_format = sl.BODY_FORMAT.BODY_38

        self.zed.enable_body_tracking(self.body_param)

        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        self.body_runtime_param.detection_confidence_threshold = 60

        camera_info = self.zed.get_camera_information()
        self.display_resolution = camera_info.camera_configuration.resolution
        self.image_scale = [
            self.display_resolution.width / camera_info.camera_configuration.resolution.width,
            self.display_resolution.height / camera_info.camera_configuration.resolution.height,
        ]

        self.bodies = sl.Bodies()
        self.image = sl.Mat()

    def grab(self) -> bool:
        """
        Grab a new frame. Returns True on success.
        """
        return self.zed.grab() == sl.ERROR_CODE.SUCCESS

    def retrieve(self):
        """
        Retrieve left image and body list after a successful grab().
        """
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT, sl.MEM.CPU, self.display_resolution)
        self.zed.retrieve_bodies(self.bodies, self.body_runtime_param)
        return self.image, self.bodies

    def close(self) -> None:
        """
        Clean up ZED resources.
        """
        self.image.free(sl.MEM.CPU)
        self.zed.disable_body_tracking()
        self.zed.disable_positional_tracking()
        self.zed.close()


# ==========================================================
# 2D viewer
# ==========================================================
class BodyViewer2D:
    """
    2D OpenCV viewer for displaying ZED body tracking results,
    the number of detected people, role labels, and a clickable
    help button.
    """

    def __init__(
        self,
        window_name: str,
        display_resolution: sl.Resolution,
        image_scale: Tuple[float, float],
        binding_manager: RoleBindingManager,
        help_image: np.ndarray,
        button_x: int = 20,
        button_y: int = 70,
        button_w: int = 170,
        button_h: int = 40,
        help_window_name: str = "Binding Help",
    ) -> None:
        self.window_name = window_name
        self.display_resolution = display_resolution
        self.image_scale = image_scale
        self.binding_manager = binding_manager

        self.button_x = button_x
        self.button_y = button_y
        self.button_w = button_w
        self.button_h = button_h

        self.help_image = help_image
        self.help_window_name = help_window_name
        self.binding_help_requested = False

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if (
                self.button_x <= x <= self.button_x + self.button_w
                and self.button_y <= y <= self.button_y + self.button_h
            ):
                self.binding_help_requested = True
                print("[UI] Binding Help button clicked.")
            else:
                self.binding_manager.register_click(x, y)

    def render_frame(self, image_left_ocv: np.ndarray, bodies: sl.Bodies, num_people: int) -> None:
        """
        Draw skeletons, people count, role labels, and help button.
        """
        cv_viewer.render_2D(
            image_left_ocv,
            self.image_scale,
            bodies.body_list,
            True,
            sl.BODY_FORMAT.BODY_38
        )

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

        # Help button background
        cv2.rectangle(
            image_left_ocv,
            (self.button_x, self.button_y),
            (self.button_x + self.button_w, self.button_y + self.button_h),
            (255, 255, 255),
            thickness=-1
        )

        # Help button border
        cv2.rectangle(
            image_left_ocv,
            (self.button_x, self.button_y),
            (self.button_x + self.button_w, self.button_y + self.button_h),
            (0, 0, 0),
            thickness=2
        )

        # Help button text
        cv2.putText(
            image_left_ocv,
            "Binding Help",
            (self.button_x + 15, self.button_y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        # Current selected role
        if self.binding_manager.current_role is not None:
            cv2.putText(
                image_left_ocv,
                f"Current role to bind: {self.binding_manager.current_role}",
                (20, self.display_resolution.height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        # Labels above each body
        for body in bodies.body_list:
            body_id = body.id

            if body.keypoint_2d is None or len(body.keypoint_2d) == 0:
                continue

            kp = body.keypoint_2d[0]
            if np.isnan(kp[0]) or np.isnan(kp[1]):
                continue

            x = int(kp[0] * self.image_scale[0])
            y = int(kp[1] * self.image_scale[1])

            role_text, color = self.binding_manager.get_label_and_color(body_id)

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

        if self.binding_help_requested:
            cv2.imshow(self.help_window_name, self.help_image)
            self.binding_help_requested = False

        cv2.imshow(self.window_name, image_left_ocv)


# ==========================================================
# Main
# ==========================================================
def main() -> None:
    print("  - Press 'q' in the main window to quit.")
    print("  - Press 'm' to pause/resume.")

    parser = build_arg_parser()
    opt = parser.parse_args()

    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both.")
        return

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    configure_init_params(init_params, opt)

    try:
        zed_tracker = ZEDBodyTracker(init_params)
    except RuntimeError as exc:
        print(exc)
        return

    binding_manager = RoleBindingManager(tuple(zed_tracker.image_scale))
    help_img = create_binding_help_image()

    window_name = "People Detection and Role Binding"
    viewer = BodyViewer2D(
        window_name=window_name,
        display_resolution=zed_tracker.display_resolution,
        image_scale=tuple(zed_tracker.image_scale),
        binding_manager=binding_manager,
        help_image=help_img,
    )

    key_wait = 10

    while True:
        if not zed_tracker.grab():
            continue

        image, bodies = zed_tracker.retrieve()
        image_left_ocv = image.get_data()
        num_people = len(bodies.body_list)

        binding_manager.try_assign_from_click(bodies)
        viewer.render_frame(image_left_ocv, bodies, num_people)

        key = cv2.waitKey(key_wait) & 0xFF

        if key == ord("q"):
            print("Exiting...")
            break

        if key == ord("m"):
            if key_wait > 0:
                print("Pause")
                key_wait = 0
            else:
                print("Restart")
                key_wait = 10

        msg = binding_manager.set_current_role_from_key(key)
        if msg:
            print(msg)

    zed_tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
