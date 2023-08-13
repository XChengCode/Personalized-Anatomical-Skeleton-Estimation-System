import json
import cv2
import numpy as np
import sys
import time
import math
import pyvista
import opensim as osim
from geometry_transformation import get_bone_length, translation_matrix_from_vectors, Vector_Estimation
from opensim.tools import InverseKinematicsTool
from sklearn.cluster import KMeans

def ReadJsonFile(path):
    with open(path,'r') as f:
        json_item = json.load(f)
        return np.array(json_item)
    
def CollateJsonFile(data):
    joint_array = np.array(data)
    Joint_array = joint_array.transpose(2,0,1)
    J_x = -Joint_array[0]
    J_y = -Joint_array[1]
    J_z = Joint_array[2]
    Joint_array = np.stack([J_x,J_y,J_z]).transpose(1,2,0)
    return Joint_array

def MOCA_Markers(path:str):
    """
    :param path: input video's path
    :return: markers group of human in ndarray format(Uing mediapipe lib)
    """
    cap = cv2.VideoCapture(path)
    frames = [] 
    timestamps = []
    startTime = time.time()
    i = 0
    while cap.isOpened():
        rep, frame = cap.read()
        if not rep:
            break
        endTime = time.time()
        i += 1
        timestamps.append(0.2*i)
        if cv2.waitKey(33) == 27:
            break
    cv2.destroyAllWindows()
    cap.release()

    return np.array(timestamps)


def Pose2Trc(points, output_path, timestamps, CameraData):
    
    with open('./pose.trc', "w") as file:
        file.truncate(0)
    
    HumanMarkerLabels = ['RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee',
                         'LAnkle', 'Neck','Nose','Head', 'LShoulder',
                         'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
    TRCfile = open(output_path+'/pose.trc','a')
    temp = sys.stdout 
    sys.stdout = TRCfile 

    
    CameraData["NumFrames"] = len(points)
    CameraData["OrigNumFrames"] = CameraData["NumFrames"] - 1

    # Writer document header
    sys.stdout.write("PathFileType\t4\t(X/Y/Z)\toutput.trc\n")
    sys.stdout.write(
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
    sys.stdout.write(
        "%d\t%d\t%d\t%d\tm\t%d\t%d\t%d\n" % (CameraData["DataRate"], CameraData["CameraRate"], CameraData["NumFrames"],
                                             CameraData["NumMarkers"], CameraData["OrigDataRate"],
                                             CameraData["OrigDataStartFrame"], CameraData["OrigNumFrames"]))

    # Write Labels
    sys.stdout.write("Frame#\tTime\t")  
    for i, label in enumerate(HumanMarkerLabels):
        if i != 0:
            sys.stdout.write("\t")
        sys.stdout.write("\t\t%s" % (label))
    sys.stdout.write("\n")
    sys.stdout.write("\t")
    for i in range(len(HumanMarkerLabels) * 3):
        sys.stdout.write("\t%c%d" % (chr(ord('X') + (i % 3)), math.ceil((i + 1) / 3)))
    sys.stdout.write("\n")

    # Write data
    for i, point in enumerate(points):
        sys.stdout.write("%d\t%f" % (i, timestamps[i]))
        for l in range(len(point)):
            if l in [0,7]: continue
            sys.stdout.write("\t%f\t%f\t%f" % (point[l][0], point[l][1], point[l][2]))
        sys.stdout.write("\n")

    sys.stdout = temp  
    print("TRC File has been saved!")



def UpdateOsimCoords(NameList, ValueList, OsimFile, output_path, idx):
    import xml.etree.ElementTree as ET
    Coordinates_Name = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
                        'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r',
                        'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l',
                        'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l','L5_S1_Flex_Ext',
                        'L5_S1_Lat_Bending', 'L5_S1_axial_rotation', 'Abs_t1', 'Abs_t2', 'neck_flexion', 'neck_bending','neck_rotation', 'arm_flex_r', 'arm_add_r',
                        'arm_rot_r', 'elbow_flex_r', 'pro_sup_r', 'wrist_flex_r', 'wrist_dev_r', 'arm_flex_l',
                        'arm_add_l', 'arm_rot_l', 'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l', 'wrist_dev_l']
    assert len(NameList) == len(ValueList)
    Parameters_Mot = {}
    for i in range(len(NameList)):
        Parameters_Mot[NameList[i]] = np.float(ValueList[i]) * np.pi / 180

    tree = ET.parse(OsimFile)
    root = tree.getroot()
    print("{}:{}".format(root.tag, root.attrib))
    Model = root.find("Model")
    print("{}:{}".format(Model.tag, Model.attrib))
    BodySet = Model.find("BodySet")
    JointSet = Model.find("JointSet")
    MarkerSet = Model.find("MarkerSet")

    JointSetObjects = JointSet.find("objects")

    # Trees  save following parts
    Trees = []
    CustomJoints = JointSetObjects.findall("CustomJoint")
    WeldJoints = JointSetObjects.findall("WeldJoint")
    PinJoints = JointSetObjects.findall("PinJoint")
    UniversalJoints = JointSetObjects.findall("UniversalJoint")
    Trees.append(CustomJoints)
    # Trees.append(WeldJoints)
    Trees.append(PinJoints)
    # Trees.append(UniversalJoints)
    for SubTrees in Trees:
        for SubTree in SubTrees:
            print(SubTree.tag)
            Coords = SubTree.find("coordinates")
            Coordinates = Coords.findall("Coordinate")
            for Coordinate in Coordinates:
                CoordName = Coordinate.attrib
                DefaultValue = Coordinate.find('default_value')
                print("ParameterName:{}, ParameterDefaultValue:{}".format(CoordName, DefaultValue.text))
                if CoordName['name'] not in Coordinates_Name:
                    continue
                if CoordName['name'] in Coordinates_Name:
                    if CoordName['name'] in ['neck_bending','mtp_angle_r','mtp_angle_l','Abs_t1','Abs_t2','pro_sup_r', 'wrist_flex_r', 'wrist_dev_r','pro_sup_l', 'wrist_flex_l', 'wrist_dev_l']: continue
                    print("Update {}'s value!-------!".format(CoordName['name']))
                    DefaultValue.text = str(Parameters_Mot[CoordName['name']])

    tree.write(output_path+'/{}.osim'.format(idx))



def mot2osim(mot_file,osim_file,output_path):
    with open(mot_file,'r') as f:
        Lines = f.readlines()
        row, col = 0,0
        NameList, ValueList = [], []
        for i,line in enumerate(Lines):
            if i==2: row = int(line.split('=')[1])
            if i==3: col = int(line.split('=')[1])
            if i == 10:
                NameList = line.split('\t')[1:]
            if i > 10:
                ValueList = line.split('\t')[1:]
                UpdateOsimCoords(NameList, ValueList,osim_file, output_path, i-11)



class Geometry:
    def __init__(self, name, body, t):
        self.name = name
        self.body = body
        self.t = t

def find_geom_by_body(name):
    BodySetList = ['pelvis','sacrum','femur_r','patella_r','tibia_r','talus_r','calcn_r',
                   'toes_r','femur_l','patella_l','tibia_l','talus_l','calcn_l','toes_l',
                   'lumbar5','lumbar4','lumbar3','lumbar2','lumbar1','torso','head','Abdomen',
                   'humerus_r','ulna_r','radius_r','hand_r','humerus_l','ulna_l','radius_l','hand_l']
    if name in BodySetList:
        return name
    return None

def PlotOsim(filename, Joint_array, idx):
    model = osim.Model(filename)
    s = model.initSystem()
    bodies = []
    meshes = []

    for i, body in enumerate(model.getBodySet()):
        if i in [0, 4, 10, 20]: 
            check_geom_string = body.getPropertyByName('attached_geometry').toString()
            mesh_file_name = 'none'
            for j in range(2):
                geom = body.get_attached_geometry(j)
                mesh_file_name = geom.getPropertyByName('mesh_file').toString()
                scale_factor = geom.getPropertyByName('scale_factors').toString().strip('()').split(' ')
                scale_Matrix = np.array([[np.float32(scale_factor[0]), 0, 0, 0],
                                         [0, np.float32(scale_factor[1]), 0, 0],
                                         [0, 0, np.float32(scale_factor[2]), 0],
                                         [0, 0, 0, 1]])
                # name = body.getName()
                name = mesh_file_name.split('.')[0]
                # print(mesh_file_name)
                p = body.getPositionInGround(s)
                r = body.getTransformInGround(s).R()
                t = np.array([[r.get(0, 0), r.get(0, 1), r.get(0, 2), p.get(0)],
                              [r.get(1, 0), r.get(1, 1), r.get(1, 2), p.get(1)],
                              [r.get(2, 0), r.get(2, 1), r.get(2, 2), p.get(2)],
                              [0, 0, 0, 1]])
                # print("name:{}".format(name), t)
                t = np.dot(t, scale_Matrix)
                mesh_geom = Geometry(name, mesh_file_name, t)
                bodies.append(mesh_geom)
        elif i in [25, 29]:  #hands
            for j in range(27):
                geom = body.get_attached_geometry(j)
                mesh_file_name = geom.getPropertyByName('mesh_file').toString()
                scale_factor = geom.getPropertyByName('scale_factors').toString().strip('()').split(' ')
                scale_Matrix = np.array([[np.float32(scale_factor[0]), 0, 0, 0],
                                         [0, np.float32(scale_factor[1]), 0, 0],
                                         [0, 0, np.float32(scale_factor[2]), 0],
                                         [0, 0, 0, 1]])

                # name = body.getName()
                name = mesh_file_name.split('.')[0]
                # print(mesh_file_name)
                p = body.getPositionInGround(s)
                r = body.getTransformInGround(s).R()
                t = np.array([[r.get(0, 0), r.get(0, 1), r.get(0, 2), p.get(0)],
                              [r.get(1, 0), r.get(1, 1), r.get(1, 2), p.get(1)],
                              [r.get(2, 0), r.get(2, 1), r.get(2, 2), p.get(2)],
                              [0, 0, 0, 1]])
                # print("name:{}".format(name), t)
                t = np.dot(t, scale_Matrix)
                mesh_geom = Geometry(name, mesh_file_name, t)
                bodies.append(mesh_geom)
        elif i in [19]:  # torso
            body_C = body.getPropertyByName('components')
            # print(body)
            for j in range(19, -1, -1):
                geom = body_C.getValueAsObject(j)  # return Object class
                # print(geom.toString()) # 'AbstractObject'
                geom_trans = geom.getPropertyByName('translation').toString().strip('()').split(' ')
                geom = geom.getPropertyByName('attached_geometry')  # return abstractObject class
                # print(geom.toString()) # '(Mesh)'
                scale_factor = geom.getValueAsObject(0).getPropertyByName('scale_factors').toString().strip('()').split(
                    ' ')
                scale_Matrix = np.array([[np.float32(scale_factor[0]), 0, 0, 0],
                                         [0, np.float32(scale_factor[1]), 0, 0],
                                         [0, 0, np.float32(scale_factor[2]), 0],
                                         [0, 0, 0, 1]])
                mesh_file_name = geom.getValueAsObject(0).getPropertyByName('mesh_file').toString()
                # print(mesh_file_name)
                # mesh_file_name = geom.getPropertyByName('mesh_file').toString()
                # name = body.getName()
                name = mesh_file_name.split('.')[0]

                t_trans = np.array([[1, 0, 0, np.float32(geom_trans[0])],
                                    [0, 1, 0, np.float32(geom_trans[1])],
                                    [0, 0, 1, np.float32(geom_trans[2])],
                                    [0, 0, 0, 1]])

                # print(t_trans)
                p = body.getPositionInGround(s)
                r = body.getTransformInGround(s).R()
                t = np.array([[r.get(0, 0), r.get(0, 1), r.get(0, 2), p.get(0)],
                              [r.get(1, 0), r.get(1, 1), r.get(1, 2), p.get(1)],
                              [r.get(2, 0), r.get(2, 1), r.get(2, 2), p.get(2)],
                              [0, 0, 0, 1]])
                # print("name:{}".format(name), t)

                t = np.dot(t_trans, np.dot(t, scale_Matrix))
                # t = np.dot(t,scale_Matrix)
                mesh_geom = Geometry(name, mesh_file_name, t)
                bodies.append(mesh_geom)
        elif i in [21]:
            print('pass')
        else:
            check_geom_string = body.getPropertyByName('attached_geometry').toString()
            # print("check_geom_string = ", check_geom_string)
            mesh_file_name = 'none'
            if check_geom_string.find('Mesh') != -1:
                geom = body.get_attached_geometry(0)
                mesh_file_name = geom.getPropertyByName('mesh_file').toString()
            # name = body.getName()
            # print(mesh_file_name)
            name = mesh_file_name.split('.')[0]
            scale_factor = geom.getPropertyByName('scale_factors').toString().strip('()').split(' ')
            scale_Matrix = np.array([[np.float32(scale_factor[0]), 0, 0, 0],
                                     [0, np.float32(scale_factor[1]), 0, 0],
                                     [0, 0, np.float32(scale_factor[2]), 0],
                                     [0, 0, 0, 1]])

            p = body.getPositionInGround(s)
            r = body.getTransformInGround(s).R()

            # Construct Transformation matrix (4x4)
            t = np.array([[r.get(0, 0), r.get(0, 1), r.get(0, 2), p.get(0)],
                          [r.get(1, 0), r.get(1, 1), r.get(1, 2), p.get(1)],
                          [r.get(2, 0), r.get(2, 1), r.get(2, 2), p.get(2)],
                          [0, 0, 0, 1]])
            # print("name:{}".format(name), t)
            t = np.dot(t, scale_Matrix)
            mesh_geom = Geometry(name, mesh_file_name, t)
            bodies.append(mesh_geom)


    transformY2Z = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    OppoY = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    OppoX = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    plotter = pyvista.Plotter(off_screen=True)
    centre_list = []
    
    for body in bodies:
        reader = pyvista.get_reader("OpenSimModel/Geometry/"+body.body.split('.')[0]+'.vtp')
        body_mesh = reader.read()
        body_mesh.transform(body.t)
        body_mesh.transform(OppoY)
        body_mesh.transform(OppoX)
        
        if body.body.split('.')[0] in ['hat_ribs_scap']: 
            left_end,right_end = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            left_end = np.array(left_end)
            right_end = np.array(right_end)
            left_point = left_end-0.11*(right_end-left_end)
            right_point = right_end-0.11*(left_end-right_end)
            bottom_point,_ = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh,Vertical=True)
            
        if body.body.split('.')[0] in ['hat_skull']: 
            skull_top, skull_bottom = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            skull_vec=np.array(skull_bottom)-np.array(skull_top)
            x, y, z = skull_vec[0],skull_vec[1],skull_vec[2]
            skull_magnitude = math.sqrt(x**2 + y**2 + z**2)
            skull_desired_length = get_bone_length(9,10,Joint_array)[0]
            scale_factor = 0.8*skull_desired_length / skull_magnitude
            skull_scale_matrix = np.array([[scale_factor,0,0,0],
                                           [0,scale_factor,0,0],
                                           [0,0,scale_factor,0],
                                           [0,0,0,1]])
        
        if body.body.split('.')[0] in ['cerv1sm']: 
            cerv1_top_point,_ = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            
        if body.body.split('.')[0] in ['cerv7']: 
            cerv6_bottom_point,_ = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            
        if body.body.split('.')[0] in ['thoracic1_s']: 
            thoracic1_top_point,_ = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
        
        
        if body.body.split('.')[0] in ['thoracic12_s']: 
            thoracic12_top_point,_ = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)

            
        if body.body.split('.')[0] in ['tibia_r']:
            head_vector,end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            r_leg_r_matrix, r_leg_s_matrix, r_leg_t_matrix = translation_matrix_from_vectors(head_vector, end_vector, 
                                                                                         Joint_array, movement,
                                                                                         body.body.split('.')[0], idx)
            body_mesh.transform(r_leg_r_matrix)
            body_mesh.transform(r_leg_s_matrix)
            body_mesh.transform(r_leg_t_matrix)
            tibia_r_head_vector,tibia_r_end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],
                                                                       body_mesh = body_mesh)
            
        if body.body.split('.')[0] in ['tibia_l']:
            head_vector,end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            l_leg_r_matrix, l_leg_s_matrix, l_leg_t_matrix = translation_matrix_from_vectors(head_vector, end_vector, Joint_array,
                                                                                             movement, body.body.split('.')[0], idx)
            body_mesh.transform(l_leg_r_matrix)
            body_mesh.transform(l_leg_s_matrix)
            body_mesh.transform(l_leg_t_matrix)
            tibia_l_head_vector,tibia_l_end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],
                                                                       body_mesh = body_mesh)
        
        if body.body.split('.')[0] in ['ulna_rv']:
            head_vector,end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            r_arm_r_matrix, r_arm_s_matrix, r_arm_t_matrix = translation_matrix_from_vectors(head_vector, end_vector, Joint_array,
                                                                                             movement,  body.body.split('.')[0], idx)
            body_mesh.transform(r_arm_r_matrix)
            body_mesh.transform(r_arm_s_matrix)
            body_mesh.transform(r_arm_t_matrix)
            ulna_r_head_vector,ulna_r_end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],
                                                                     body_mesh = body_mesh)
            
        if body.body.split('.')[0] in ['ulna_lv']:
            head_vector,end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            l_arm_r_matrix, l_arm_s_matrix, l_arm_t_matrix = translation_matrix_from_vectors(head_vector, end_vector, Joint_array,
                                                                                             movement, body.body.split('.')[0], idx)
            body_mesh.transform(l_arm_r_matrix)
            body_mesh.transform(l_arm_s_matrix)
            body_mesh.transform(l_arm_t_matrix)
            ulna_l_head_vector,ulna_l_end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],
                                                                     body_mesh = body_mesh)
            
        if body.body.split('.')[0] in ['femur_r']: 
            head_vector,end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            femur_r_r_matrix, femur_r_s_matrix, femur_r_t_matrix = translation_matrix_from_vectors(head_vector, end_vector, Joint_array,
                                                                                                   movement, body.body.split('.')[0], idx)
        if body.body.split('.')[0] in ['femur_l']: 
            head_vector,end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            femur_l_r_matrix, femur_l_s_matrix, femur_l_t_matrix = translation_matrix_from_vectors(head_vector, end_vector, Joint_array,
                                                                                                   movement, body.body.split('.')[0], idx)
        if body.body.split('.')[0] in ['humerus_rv']: 
            head_vector,end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            humerus_r_r_matrix, humerus_r_s_matrix, humerus_r_t_matrix = translation_matrix_from_vectors(head_vector, end_vector, Joint_array,
                                                                                                   movement, body.body.split('.')[0], idx)
        if body.body.split('.')[0] in ['humerus_lv']: 
            head_vector,end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            humerus_l_r_matrix, humerus_l_s_matrix, humerus_l_t_matrix = translation_matrix_from_vectors(head_vector, end_vector, Joint_array,
                                                                                                   movement, body.body.split('.')[0], idx)
            
        if body.body.split('.')[0] in ['l_pelvis', 'r_pelvis']: 
            centre_list.append(np.array(body_mesh.center))
            if(len(centre_list)==2):
                center = (centre_list[0]+centre_list[1])/2
                movement = Joint_array[idx][0]-center
                movement_matrix = np.array([[1,0,0, movement[0]],
                                            [0,1,0, movement[1]],
                                            [0,0,1, movement[2]],
                                            [0, 0, 0, 1]])
            if body.body.split('.')[0] in ['l_pelvis']:
                left_side_vector,_ = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            if body.body.split('.')[0] in ['r_pelvis']:
                right_side_vector,_ = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)

    body_movement = np.array(thoracic12_top_point) - np.array(bottom_point)
    body_movement_matrix = np.array([[1,0,0,body_movement[0]],
                                     [0,1,0,body_movement[1]],
                                     [0,0,1,body_movement[2]],
                                     [0,0,0,1]]) 
        
    neck_movement = np.array(thoracic1_top_point) - np.array(cerv6_bottom_point)
    neck_movement_matrix = np.array([[1,0,0,neck_movement[0]],
                                     [0,1,0,neck_movement[1]],
                                     [0,0,1,neck_movement[2]],
                                     [0,0,0,1]]) 
    
    pel_left_point = left_side_vector-0.11*(np.array(right_side_vector)-np.array(left_side_vector))
    pel_right_point = right_side_vector-0.11*(np.array(left_side_vector)-np.array(right_side_vector))
    
    for body in bodies:
        if body.body.split('.')[0] in ['cerv1sm','cerv2sm','cerv3sm','cerv4sm','cerv5sm','cerv6sm','cerv7']:
            body_mesh.transform(neck_movement_matrix)
            if body.body.split('.')[0] in ['cerv1sm']: 
                cerv1_top_point,_ = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
       
        reader = pyvista.get_reader("OpenSimModel/Geometry/"+body.body.split('.')[0]+'.vtp')
        body_mesh = reader.read()
        body_mesh.transform(body.t)
        body_mesh.transform(OppoY)
        body_mesh.transform(OppoX) 
        
        if body.body.split('.')[0] in ['hat_skull']:
            #body_mesh.transform(skull_scale_matrix)
            new_skull_top, new_skull_bottom = Vector_Estimation(bone_name = body.body.split('.')[0],
                                                                body_mesh = body_mesh)
                
        if body.body.split('.')[0] in ['femur_l']:
            body_mesh.transform(femur_l_r_matrix)
            body_mesh.transform(femur_l_s_matrix)
            body_mesh.transform(femur_l_t_matrix)
            new_head_vector, new_end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            translate_factor = np.array(new_end_vector) - np.array(tibia_l_head_vector)
            tibia_l_translation_matrix = np.array([[1,0,0,translate_factor[0]],
                                                  [0,1,0,translate_factor[1]],
                                                  [0,0,1,translate_factor[2]],
                                                  [0,0,0,1]])
            
            move_2_pel_factor = np.array(pel_left_point) - np.array(new_head_vector)
            pel_l_translation_matrix = np.array([[1,0,0,move_2_pel_factor[0]],
                                                  [0,1,0,move_2_pel_factor[1]],
                                                  [0,0,1,move_2_pel_factor[2]],
                                                  [0,0,0,1]])
            
                
        if body.body.split('.')[0] in ['femur_r']:
            body_mesh.transform(femur_r_r_matrix)
            body_mesh.transform(femur_r_s_matrix)
            body_mesh.transform(femur_r_t_matrix)
            new_head_vector, new_end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            translate_factor = np.array(new_end_vector) - np.array(tibia_r_head_vector)
            tibia_r_translation_matrix = np.array([[1,0,0,translate_factor[0]],
                                                  [0,1,0,translate_factor[1]],
                                                  [0,0,1,translate_factor[2]],
                                                  [0,0,0,1]])
            
            move_2_pel_factor = np.array(pel_right_point) - np.array(new_head_vector)
            pel_r_translation_matrix = np.array([[1,0,0,move_2_pel_factor[0]],
                                                  [0,1,0,move_2_pel_factor[1]],
                                                  [0,0,1,move_2_pel_factor[2]],
                                                  [0,0,0,1]])
            
            
        if body.body.split('.')[0] in ['humerus_lv']:
            body_mesh.transform(humerus_l_r_matrix)
            body_mesh.transform(humerus_l_s_matrix)
            body_mesh.transform(humerus_l_t_matrix)
            new_head_vector, new_end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            translate_factor = np.array(left_point) - np.array(new_head_vector)
            humerus_l_translation_matrix = np.array([[1,0,0,translate_factor[0]],
                                                  [0,1,0,translate_factor[1]],
                                                  [0,0,1,translate_factor[2]],
                                                  [0,0,0,1]])
            body_mesh.transform(humerus_l_translation_matrix)
            body_mesh.transform(body_movement_matrix)
            new_head_vector, new_end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            translate_factor = np.array(new_end_vector) - np.array(ulna_l_head_vector)
            ulna_l_translation_matrix = np.array([[1,0,0,translate_factor[0]],
                                                  [0,1,0,translate_factor[1]],
                                                  [0,0,1,translate_factor[2]],
                                                  [0,0,0,1]])
                
        if body.body.split('.')[0] in ['humerus_rv']:
            body_mesh.transform(humerus_r_r_matrix)
            body_mesh.transform(humerus_r_s_matrix)
            body_mesh.transform(humerus_r_t_matrix)
            new_head_vector, _ = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            translate_factor = np.array(right_point) - np.array(new_head_vector)
            humerus_r_translation_matrix = np.array([[1,0,0,translate_factor[0]],
                                                  [0,1,0,translate_factor[1]],
                                                  [0,0,1,translate_factor[2]],
                                                  [0,0,0,1]]) 
            body_mesh.transform(humerus_r_translation_matrix)
            body_mesh.transform(body_movement_matrix)
            new_head_vector, new_end_vector = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            translate_factor = np.array(new_end_vector) - np.array(ulna_r_head_vector)
            ulna_r_translation_matrix = np.array([[1,0,0,translate_factor[0]],
                                                  [0,1,0,translate_factor[1]],
                                                  [0,0,1,translate_factor[2]],
                                                  [0,0,0,1]])
            
    skull_translate_factor = np.array(cerv1_top_point) - np.array(new_skull_bottom)
    skull_translation_matrix = np.array([[1,0,0,skull_translate_factor[0]],
                                                  [0,1,0,skull_translate_factor[1]],
                                                  [0,0,1,skull_translate_factor[2]],
                                                  [0,0,0,1]])
    angle = np.radians(0)
    skull_rotate_matrix = np.array([[np.cos(angle), 0, np.sin(angle), 0],
                                  [0, 1, 0, 0],
                                  [-np.sin(angle), 0, np.cos(angle), 0],
                                     [0, 0, 0, 1 ]  ]) 
    
    for body in bodies:
        reader = pyvista.get_reader("OpenSimModel/Geometry/"+body.body.split('.')[0]+'.vtp')
        body_mesh = reader.read()
        if body.body.split('.')[0] in ['hat_skull','hat_jaw']:
            body_mesh.transform(skull_rotate_matrix)
        body_mesh.transform(body.t)
        body_mesh.transform(OppoY)
        body_mesh.transform(OppoX)
        
        if body.body.split('.')[0] in ['hat_skull','hat_jaw']:
            #body_mesh.transform(skull_scale_matrix)
            body_mesh.transform(skull_translation_matrix)

        if body.body.split('.')[0] in ['cerv1sm','cerv2sm','cerv3sm','cerv4sm',
                                       'cerv5sm','cerv6sm','cerv7']:
            body_mesh.transform(neck_movement_matrix)
            
        if body.body.split('.')[0] in ['hat_ribs_scap']:  
            body_mesh.transform(body_movement_matrix)
            
        if body.body.split('.')[0] in ['tibia_r','fibula_r','talus_rv','foot','bofoot']:
            body_mesh.transform(r_leg_r_matrix)
            body_mesh.transform(r_leg_s_matrix)
            body_mesh.transform(r_leg_t_matrix)
            body_mesh.transform(tibia_r_translation_matrix)
            body_mesh.transform(pel_r_translation_matrix)
            if body.body.split('.')[0] in ['tibia_r']:
                _, joint3 = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
                
        if body.body.split('.')[0] in ['tibia_l','fibula_l','talus_lv','l_foot','l_bofoot']:
            body_mesh.transform(l_leg_r_matrix)
            body_mesh.transform(l_leg_s_matrix)
            body_mesh.transform(l_leg_t_matrix) 
            body_mesh.transform(tibia_l_translation_matrix)
            body_mesh.transform(pel_l_translation_matrix)
            if body.body.split('.')[0] in ['tibia_l']:
                _, joint6 = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            
        if body.body.split('.')[0] in ['ulna_rv','radius_rv','pisiform_rvs','lunate_rvs','scaphoid_rvs',
                                       'triquetrum_rvs','hamate_rvs','capitate_rvs','trapezoid_rvs',
                                       'trapezium_rvs','metacarpal2_rvs','index_proximal_rvs','index_medial_rvs',
                                       'index_distal_rvs','metacarpal3_rvs','middle_proximal_rvs',
                                       'middle_medial_rvs','middle_distal_rvs','metacarpal4_rvs',
                                       'ring_proximal_rvs','ring_medial_rvs','ring_distal_rvs','metacarpal5_rvs',
                                       'little_proximal_rvs','little_medial_rvs','little_distal_rvs',
                                       'metacarpal1_rvs','thumb_proximal_rvs','thumb_distal_rvs']:
            body_mesh.transform(r_arm_r_matrix)
            body_mesh.transform(r_arm_s_matrix)
            body_mesh.transform(r_arm_t_matrix)
            body_mesh.transform(ulna_r_translation_matrix)
            if body.body.split('.')[0] in ['ulna_rv']:
                _, joint16 = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
                
        if body.body.split('.')[0] in ['ulna_lv','radius_lv','pisiform_lvs','lunate_lvs','scaphoid_lvs',
                                       'triquetrum_lvs','hamate_lvs','capitate_lvs','trapezoid_lvs',
                                       'trapezium_lvs','metacarpal2_lvs','index_proximal_lvs','index_medial_lvs',
                                       'index_distal_lvs','metacarpal3_lvs','middle_proximal_lvs',
                                       'middle_medial_lvs','middle_distal_lvs','metacarpal4_lvs',
                                       'ring_proximal_lvs','ring_medial_lvs','ring_distal_lvs','metacarpal5_lvs',
                                       'little_proximal_lvs','little_medial_lvs','little_distal_lvs',
                                       'metacarpal1_lvs','thumb_proximal_lvs','thumb_distal_lvs']:
            body_mesh.transform(l_arm_r_matrix)
            body_mesh.transform(l_arm_s_matrix)
            body_mesh.transform(l_arm_t_matrix)
            body_mesh.transform(ulna_l_translation_matrix)
            if body.body.split('.')[0] in ['ulna_lv']:
                _, joint13 = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
        
        if body.body.split('.')[0] in ['r_patella']: 
            body_mesh.transform(femur_r_r_matrix)
            body_mesh.transform(femur_r_s_matrix)
            body_mesh.transform(femur_r_t_matrix)
            body_mesh.transform(pel_r_translation_matrix)
            
        if body.body.split('.')[0] in ['l_patella']: 
            body_mesh.transform(femur_l_r_matrix)
            body_mesh.transform(femur_l_s_matrix)
            body_mesh.transform(femur_l_t_matrix)
            body_mesh.transform(pel_l_translation_matrix)
            
        if body.body.split('.')[0] in ['humerus_rv','femur_r','femur_l','humerus_lv']: 
          
            if body.body.split('.')[0] in ['femur_l']:
                body_mesh.transform(femur_l_r_matrix)
                body_mesh.transform(femur_l_s_matrix)
                body_mesh.transform(femur_l_t_matrix)
                body_mesh.transform(pel_l_translation_matrix)
                joint4, joint5 = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
  
            if body.body.split('.')[0] in ['femur_r']:
                body_mesh.transform(femur_r_r_matrix)
                body_mesh.transform(femur_r_s_matrix)
                body_mesh.transform(femur_r_t_matrix)
                body_mesh.transform(pel_r_translation_matrix)
                joint1, joint2 = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            
            if body.body.split('.')[0] in ['humerus_lv']:
                body_mesh.transform(humerus_l_r_matrix)
                body_mesh.transform(humerus_l_s_matrix)
                body_mesh.transform(humerus_l_t_matrix)
                body_mesh.transform(humerus_l_translation_matrix)
                body_mesh.transform(body_movement_matrix)
                joint11, joint12 = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
                
            if body.body.split('.')[0] in ['humerus_rv']:
                body_mesh.transform(humerus_r_r_matrix)
                body_mesh.transform(humerus_r_s_matrix)
                body_mesh.transform(humerus_r_t_matrix)
                body_mesh.transform(humerus_r_translation_matrix)
                body_mesh.transform(body_movement_matrix)
                joint14, joint15 = Vector_Estimation(bone_name = body.body.split('.')[0],body_mesh = body_mesh)
            
          
        body_mesh.transform(transformY2Z)    
        plotter.add_mesh(body_mesh,opacity="sigmoid")
    plotter.camera_position = 'xy'
    plotter.background_color = pyvista.Color("#00000000")
    #plotter.camera.position = (2.5,-0.1,5)
    plotter.screenshot("./image_folder/"+"{}s.png".format(idx),transparent_background=True)
    #plotter.show(screenshot="./Images/"+"{}.jpg".format(idx))
