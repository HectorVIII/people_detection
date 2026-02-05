import argparse
import pyzed.sl as sl


def configure_init_params(init: sl.InitParameters, opt: argparse.Namespace) -> None:
    """
    Parse command line options and configure ZED InitParameters.
    This mirrors the original parse_args() logic.
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
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
    return parser

