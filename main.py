"""
Entry point script wiring together:
  - CLI argument parsing
  - ZED body tracking wrapper
  - Role binding manager
  - 2D OpenCV viewer with "Binding Help" button
"""
import cv2
import pyzed.sl as sl

from args_parser import build_arg_parser, configure_init_params
from help_window import create_binding_help_image
from binding_manager import RoleBindingManager
from zed_body_tracking import ZEDBodyTracker
from viewer_2d import BodyViewer2D


def main():
    print("  - Press 'q' in the main window to quit.")
    print("  - Press 'm' to pause/resume.")

    # ----------------------------------------------------------
    # 1) Parse CLI arguments
    # ----------------------------------------------------------
    parser = build_arg_parser()
    opt = parser.parse_args()

    # Safety check: do not specify both SVO and IP at the same time
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        return

    # ----------------------------------------------------------
    # 2) Prepare ZED InitParameters (default + CLI configuration)
    # ----------------------------------------------------------
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    configure_init_params(init_params, opt)

    # ----------------------------------------------------------
    # 3) Initialize ZEDBodyTracker
    # ----------------------------------------------------------
    try:
        zed_tracker = ZEDBodyTracker(init_params)
    except RuntimeError as exc:
        print(exc)
        return

    # ----------------------------------------------------------
    # 4) Set up role binding + help image + viewer
    # ----------------------------------------------------------
    binding_manager = RoleBindingManager(tuple(zed_tracker.image_scale))
    help_img = create_binding_help_image()  # Create help image

    window_name = "People Detection and Role Binding"
    viewer = BodyViewer2D(
        window_name=window_name,
        display_resolution=zed_tracker.display_resolution,
        image_scale=tuple(zed_tracker.image_scale),
        binding_manager=binding_manager,
        help_image=help_img,
    )

    key_wait = 10

    # ----------------------------------------------------------
    # 5) Main loop
    # ----------------------------------------------------------
    while True:
        if not zed_tracker.grab():
            continue

        image, bodies = zed_tracker.retrieve()
        image_left_ocv = image.get_data()
        num_people = len(bodies.body_list)

        # Try to assign roles based on last click (if any)
        binding_manager.try_assign_from_click(bodies)

        # Render frame
        viewer.render_frame(image_left_ocv, bodies, num_people)

        # Keyboard handling
        key = cv2.waitKey(key_wait) & 0xFF

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

        # Role selection via keyboard
        msg = binding_manager.set_current_role_from_key(key)
        if msg:
            print(msg)

    # ----------------------------------------------------------
    # 6) Cleanup
    # ----------------------------------------------------------
    zed_tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

