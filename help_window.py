import cv2
import numpy as np


def create_binding_help_image(width: int = 700, height: int = 600) -> np.ndarray:
    """
    Create a black image with white English instructions
    explaining how to bind people to roles.
    """
    help_img = np.zeros((height, width, 3), dtype=np.uint8)  # Black background

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
    ]

    y0 = 30  # Initial y position
    dy = 24  # Line spacing

    for i, line in enumerate(lines):
        y = y0 + i * dy  # Calculate y position for this line
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

