import sys
import time
from pathlib import Path


VICON_SDK_PATH = Path(r"D:\ViconDataStream\Win64\Python\vicon_dssdk")
if VICON_SDK_PATH.exists():
    sys.path.insert(0, str(VICON_SDK_PATH))

from vicon_dssdk import ViconDataStream


VICON_HOST = "192.168.30.105"  # Vicon server IP
SUBJECT_NAME = "kedaya9"         # Rigid body / subject name in Vicon


client = ViconDataStream.Client()

print(f"Connecting to Vicon server: {VICON_HOST} ...")

client.SetConnectionTimeout(1000)
while not client.IsConnected():
    try:
        client.Connect(VICON_HOST)
    except ViconDataStream.DataStreamException as exception:
        print("Connect failed:", exception)
        time.sleep(1)

print("Connected!")

client.EnableSegmentData()
client.SetStreamMode(ViconDataStream.Client.StreamMode.EClientPull)
client.SetAxisMapping(
    ViconDataStream.Client.AxisMapping.EForward,
    ViconDataStream.Client.AxisMapping.ELeft,
    ViconDataStream.Client.AxisMapping.EUp,
)

while True:
    try:
        has_frame = client.GetFrame()
    except ViconDataStream.DataStreamException as exception:
        print("GetFrame failed:", exception)
        time.sleep(0.01)
        continue

    if not has_frame:
        time.sleep(0.01)
        continue

    frame_number = client.GetFrameNumber()
    root_segment = client.GetSubjectRootSegmentName(SUBJECT_NAME)

    translation, translation_occluded = client.GetSegmentGlobalTranslation(
        SUBJECT_NAME, root_segment
    )
    rotation, rotation_occluded = client.GetSegmentGlobalRotationEulerXYZ(
        SUBJECT_NAME, root_segment
    )

    if translation_occluded or rotation_occluded:
        print(f"Frame {frame_number}: {SUBJECT_NAME} occluded")
    else:
        x, y, z = translation
        rx, ry, rz = rotation

        print(
            f"Frame {frame_number} | "
            f"{SUBJECT_NAME}: "
            f"x={x:.1f} mm, y={y:.1f} mm, z={z:.1f} mm, "
            f"rx={rx:.3f}, ry={ry:.3f}, rz={rz:.3f}"
        )

    time.sleep(0.01)
