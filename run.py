import os
from Pose_to_Skeleton import *
from geometry_transformation import *

if __name__ == "__main__":
    CameraData = {}
    CameraData["DataRate"] = 60
    CameraData["CameraRate"] = 60
    CameraData["NumFrames"] = -1
    CameraData["NumMarkers"] = 15 
    CameraData["Units"] = 'm' # 
    CameraData["OrigDataRate"] = 60
    CameraData["OrigDataStartFrame"] = 0
    CameraData["OrigNumFrames"] = -1

    input_video_name = 'sample_video.mp4'
    video_path = './test_videos/' + input_video_name
    json_path = 'Landmarks/landmarkers.json'

    LandMarkers = ReadJsonFile(json_path)
    collated_landmarkers = CollateJsonFile(LandMarkers)
    output_path = './'
    timestamps = MOCA_Markers(video_path)
    Pose2Trc(LandMarkers, output_path, timestamps, CameraData)

    trc_file = "./pose.trc"
    trcxml_file = "./IK_Setup_Pose2Sim_Body25b.xml"
    osim_file = "./Model_Pose2Sim_Body25b.osim"

    model = osim.Model(osim_file)
    ik_set = osim.IKTaskSet()
    ik_tool = InverseKinematicsTool(trcxml_file)
    ik_tool.setModel(model)
    ik_tool.setMarkerDataFileName(trc_file)
    ik_tool.run()

    OsimFolders_Path = "./OsimFiles"
    mot_file = "./pose.mot"
    mot2osim(mot_file, osim_file, OsimFolders_Path)

    OsimFolder_Path = "./OsimFiles"
    Files = os.listdir(OsimFolder_Path)
    for i in range(1): #modify here to select the number of output frames
        filename = "{}.osim".format(i)
        PlotOsim(filename =os.path.join(OsimFolder_Path,filename), Joint_array = collated_landmarkers, idx = i)
