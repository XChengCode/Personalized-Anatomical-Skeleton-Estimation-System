import numpy as np
import math
from sklearn.cluster import KMeans

def one_d_kmeans(data, k, max_iterations=10):
    num_data_points = len(data)
    centroids = np.random.choice(data, k, replace=False)
    for _ in range(max_iterations):
        labels = np.argmin(np.abs(data[:, np.newaxis] - centroids), axis=1)
        new_centroids = np.array([data[labels == i].mean() for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids

def get_bone_length(head_joint=None, end_joint = None, Joint_array = None):
    length_list=[]
    for frame in Joint_array:
        Point_1 = frame[head_joint]
        Point_2 = frame[end_joint]
        bone_length = np.sqrt((Point_1[0]-Point_2[0])**2+(Point_1[1]-Point_2[1])**2+(Point_1[2]-Point_2[2])**2)
        length_list.append(bone_length)
    length_array = np.array(length_list)
    centroids = one_d_kmeans(length_array, 1)
    
    return centroids


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
 
    else:
        return np.eye(3)  # cross of all zeros only occurs on identical directions
    
def scale_vector(vec1=None, vec2=None, head_joint=None, end_joint = None, Joint_array=None):
    x, y, z = vec1[0],vec1[1],vec1[2]
    a, b, c = vec2[0],vec2[1],vec2[2]
    magnitude = math.sqrt(x**2 + y**2 + z**2)
    desired_length = get_bone_length(head_joint,end_joint, Joint_array)[0]
    scale_factor = desired_length / magnitude
    scale_matrix = np.array([[scale_factor,0,0,0],
                             [0,scale_factor,0,0],
                             [0,0,scale_factor,0],
                             [0,0,0,1]])
    return scale_matrix

def translation_matrix_from_vectors(vec0=None, vec1=None, Joint_array=None, movement=[0,0,0], bone_name=None, frame_num=None):
    
    if(bone_name=='hat_ribs_scap'):
        head_joint,end_joint = 7,8
    if(bone_name=='humerus_lv'):
        head_joint,end_joint = 11,12
    if(bone_name=='humerus_rv'):
        head_joint,end_joint = 14,15
    if(bone_name=='ulna_lv'):
        head_joint,end_joint = 12,13
    if(bone_name=='ulna_rv'):
        head_joint,end_joint = 15,16
    if(bone_name=='femur_l'):
        head_joint,end_joint = 4,5
    if(bone_name=='femur_r'):
        head_joint,end_joint = 1,2
    if(bone_name=='tibia_l'):
        head_joint,end_joint = 5,6
    if(bone_name=='tibia_r'):
        head_joint,end_joint = 2,3
    
    vec_0_d3 = np.array([ vec0[0],vec0[1], vec0[2]])
    vec_1_d3 = np.array([ vec1[0],vec1[1], vec1[2]])
    
    Vector_0 = vec_1_d3-vec_0_d3
    Vector_1 = Joint_array[frame_num][end_joint]-Joint_array[frame_num][head_joint]
    
    rotation_matrix = rotation_matrix_from_vectors(Vector_0, Vector_1)
    rotation_matrix = np.array([[rotation_matrix[0,0],rotation_matrix[0,1],rotation_matrix[0,2],0],
                                [rotation_matrix[1,0],rotation_matrix[1,1],rotation_matrix[1,2],0],
                                [rotation_matrix[2,0],rotation_matrix[2,1],rotation_matrix[2,2],0],
                                [0,0,0,1]])
    
    vec_0_d4 = np.array([ vec0[0],vec0[1], vec0[2], 1])
    vec_1_d4 = np.array([ vec1[0],vec1[1], vec1[2], 1])
    
    rotated_vec_0 = np.dot(rotation_matrix,vec_0_d4)
    rotated_vec_1 = np.dot(rotation_matrix,vec_1_d4)
    
    scale_matrix = scale_vector(rotated_vec_1-rotated_vec_0, Vector_1, head_joint, end_joint, Joint_array)
    
    scaled_vec_0 = np.dot(scale_matrix,rotated_vec_0)
    scaled_vec_1 = np.dot(scale_matrix,rotated_vec_1)
    
    translation_matrix = np.array([[1,0,0,-(scaled_vec_0[:3]-Joint_array[frame_num][head_joint]+movement[0])[0]],
                                   [0,1,0,-(scaled_vec_0[:3]-Joint_array[frame_num][head_joint]+movement[1])[1]],
                                   [0,0,1,-(scaled_vec_0[:3]-Joint_array[frame_num][head_joint]+movement[2])[2]],
                                   [0,0,0,1]])
    
    return rotation_matrix,scale_matrix,translation_matrix



def Vector_Estimation(bone_name=None,body_mesh=None,Vertical=False):

    #bone_name = body.body.split('.')[0]
    bone_bounds = body_mesh.bounds
    body_mesh_points = body_mesh.points
    body_mesh_points = np.array(body_mesh_points)
    kmeans_model = KMeans(n_clusters=2).fit(body_mesh_points)

    edge_0 = np.bincount(kmeans_model.labels_)[0]
    edge_1 = np.bincount(kmeans_model.labels_)[1]
    
    
    if bone_name in ['hat_ribs_scap']:
        if(Vertical==False):
            vec0 = [body_mesh_points[3960][0],body_mesh_points[3960][1],body_mesh_points[3960][2]]
            vec1 = [body_mesh_points[4717][0],body_mesh_points[4717][1],body_mesh_points[4717][2]]
        else:
            vec0 = [((body_mesh_points[2206]+body_mesh_points[729])/2)[0],
                    ((body_mesh_points[2206]+body_mesh_points[729])/2)[1],
                    ((body_mesh_points[2206]+body_mesh_points[729])/2)[2]]
            vec1 = 0
    elif bone_name in ['thoracic1_s']:
        vec0 = [body_mesh_points[110][0],body_mesh_points[110][1],body_mesh_points[110][2]]
        vec1 = 0
    elif bone_name in ['thoracic12_s']:
        vec0 = [body_mesh_points[110][0],body_mesh_points[110][1],body_mesh_points[110][2]]
        vec1 = 0
    elif bone_name in ['cerv1sm']:
        vec0 = [((body_mesh_points[105]+body_mesh_points[117])/2)[0],
                ((body_mesh_points[105]+body_mesh_points[117])/2)[1],
                ((body_mesh_points[105]+body_mesh_points[117])/2)[2]]
        vec1 = 0
    elif bone_name in ['cerv7']:
        vec0 = [body_mesh_points[2][0],body_mesh_points[2][1],body_mesh_points[2][2]]
        vec1 = 0
    elif bone_name in ['hat_skull']:
        vec0 = [body_mesh_points[2690][0],body_mesh_points[2690][1],body_mesh_points[2690][2]]
        vec1 = [body_mesh_points[690][0],body_mesh_points[690][1],body_mesh_points[690][2]]
    elif bone_name in ['r_pelvis','l_pelvis']:
        vec0 = [body_mesh_points[103][0],body_mesh_points[103][1],body_mesh_points[103][2]]
        vec1 = 0
    elif bone_name in ['femur_r','femur_l']:
        vec0 = [body_mesh_points[0][0],body_mesh_points[0][1],body_mesh_points[0][2]]
        vec1 = [body_mesh_points[-3][0],body_mesh_points[-3][1],body_mesh_points[-3][2]]
    elif bone_name in ['tibia_r','tibia_l']:
        vec0 = [body_mesh_points[1][0],body_mesh_points[1][1],body_mesh_points[1][2]]
        vec1 = [body_mesh_points[-1][0],body_mesh_points[-1][1],body_mesh_points[-1][2]]
    elif bone_name in ['humerus_rv','humerus_lv']:
        vec0 = [body_mesh_points[207][0],body_mesh_points[207][1],body_mesh_points[207][2]]
        vec1 = [body_mesh_points[-11][0],body_mesh_points[-11][1],body_mesh_points[-11][2]]
    else:
        vec0 = [body_mesh_points[-3][0],body_mesh_points[-3][1],body_mesh_points[-3][2]]
        vec1 = [body_mesh_points[0][0],body_mesh_points[0][1],body_mesh_points[0][2]]
        
    return vec0,vec1

