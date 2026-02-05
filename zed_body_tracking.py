from __future__ import annotations
from typing import Tuple
import pyzed.sl as sl


class ZEDBodyTracker:
    """
    Thin wrapper around ZED SDK body tracking configuration and frame retrieval.
    """

    def __init__(self, init_params: sl.InitParameters) -> None:
        self.zed = sl.Camera()
        self.init_params = init_params

        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Camera open error: {err}")

        # Enable positional tracking
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        positional_tracking_parameters.set_as_static = True
        self.zed.enable_positional_tracking(positional_tracking_parameters)

        # Configure body tracking parameters
        self.body_param = sl.BodyTrackingParameters()
        self.body_param.enable_tracking = True
        self.body_param.enable_body_fitting = True
        self.body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        self.body_param.body_format = sl.BODY_FORMAT.BODY_38

        # Enable body tracking module
        self.zed.enable_body_tracking(self.body_param)

        # Runtime parameters
        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        self.body_runtime_param.detection_confidence_threshold = 70

        # Camera info and display scaling
        camera_info = self.zed.get_camera_information()
        self.display_resolution = camera_info.camera_configuration.resolution
        self.image_scale = [
            self.display_resolution.width / camera_info.camera_configuration.resolution.width,
            self.display_resolution.height / camera_info.camera_configuration.resolution.height,
        ]

        # ZED containers
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

