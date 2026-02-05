from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np
import cv_viewer.tracking_viewer as cv_viewer
import pyzed.sl as sl

from binding_manager import RoleBindingManager


class BodyViewer2D:
    """
    2D OpenCV viewer for displaying ZED body tracking results,
    the number of detected people, role labels, and a clickable
    "Binding Help" button that opens a separate window.
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

    # -----------------------------
    # Mouse handling
    # -----------------------------
    def _mouse_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click inside help button
            if (self.button_x <= x <= self.button_x + self.button_w and
                    self.button_y <= y <= self.button_y + self.button_h):
                self.binding_help_requested = True
                print("[UI] Binding Help button clicked.")
            else:
                # Forward to RoleBindingManager
                self.binding_manager.register_click(x, y)

    # -----------------------------
    # Rendering
    # -----------------------------
    def render_frame(self, image_left_ocv: np.ndarray, bodies: sl.Bodies, num_people: int) -> None:
        """
        Draw skeletons, people count, role labels, and help button, then show the window.
        """
        # Draw skeletons via helper
        cv_viewer.render_2D(
            image_left_ocv,
            self.image_scale,
            bodies.body_list,
            True,  # enable_tracking - always True here
            sl.BODY_FORMAT.BODY_38  # body format (matching body_param)
        )

        # Draw total number of people
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

        # Draw "Binding Help" button
        # background
        cv2.rectangle(
            image_left_ocv,
            (self.button_x, self.button_y),
            (self.button_x + self.button_w, self.button_y + self.button_h),
            (255, 255, 255),
            thickness=-1
        )
        # border
        cv2.rectangle(
            image_left_ocv,
            (self.button_x, self.button_y),
            (self.button_x + self.button_w, self.button_y + self.button_h),
            (0, 0, 0),
            thickness=2
        )
        # text
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

        # Show current selected role at bottom-left
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

        # Draw label for each body (ID or assigned role)
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

        # Show help window if requested
        if self.binding_help_requested:
            cv2.imshow(self.help_window_name, self.help_image)
            self.binding_help_requested = False

        # Show main image
        cv2.imshow(self.window_name, image_left_ocv)

