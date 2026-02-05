from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np
import pyzed.sl as sl


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
        "Surgeon": (0, 0, 255),    # RED
        "Assistant": (255, 0, 0),  # BLUE
        "Nurse": (0, 255, 0)       # GREEN
    })

    def set_current_role_from_key(self, key: int) -> Optional[str]:
        """
        Map keyboard key to a role. Returns a small status string for logging.
        """
        if key == ord('s'):
            self.current_role = "Surgeon"
            return "[Role] Current role set to 'Surgeon'. Click on a person to bind."
        if key == ord('a'):
            self.current_role = "Assistant"
            return "[Role] Current role set to 'Assistant'. Click on a person to bind."
        if key == ord('n'):
            self.current_role = "Nurse"
            return "[Role] Current role set to 'Nurse'. Click on a person to bind."
        if key == 27:  # ESC
            self.current_role = None
            return "[Role] Binding cancelled."
        return None

    def register_click(self, x: int, y: int) -> None:
        """
        Called by the mouse callback when the user clicks in the viewer
        (outside of the help button).
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
        # Fallback: should not happen
        return self.current_role or "Unknown"

    def try_assign_from_click(self, bodies: sl.Bodies) -> None:
        """
        If a click is pending and a role is selected, find the nearest body
        to the click position (in display coordinates) and assign the role.
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

        # Reset click flag (one click = one assignment)
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
            color = (0, 255, 255)  # Yellow for raw ID (no role)

        return role_text, color

