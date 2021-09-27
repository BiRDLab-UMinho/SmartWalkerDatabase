
import cv2
import numpy as np
import matplotlib.pyplot as plt


##### metadata #####
# original frames are 640x480 pixels for img and depth
frames_original_shape = (640, 480)

# internal calibration is always the same
default_cam_intrinsics = dict(
        gait=np.array(
                [[519.429, 0.00000, 313.237, 0.0],
                 [0.00000, 519.429, 239.934, 0.0],
                 [0.00000, 0.00000, 1.00000, 0.0],
                 [0.00000, 0.00000, 0.00000, 1.0]]),

        posture=np.array(
                [[514.283, 0.00000, 315.242, 0.0],
                 [0.00000, 514.283, 245.109, 0.0],
                 [0.00000, 0.00000, 1.00000, 0.0],
                 [0.00000, 0.00000, 0.00000, 1.0]])
)

# names for each of the processed xsens skeleton joints
xsens_joint_names = (
        "pelvis", "l5", "l3", "t12", "t8",                                                  # 0-5
        "neck", "head", "right_shoulder", "right_upper_arm", "right_forearm",               # 5-10
        "right_hand", "left_shoulder", "left_upper_arm", "left_forearm", "left_hand",       # 10-15
        "right_upper_leg", "right_lower_leg", "right_foot", "right_toe", "left_upper_leg",  # 15-20
        "left_lower_leg", "left_foot", "left_toe"                                           # 20-23
)


##### common rotation matrices #####
rotx90 = np.array([[1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [0.0, 1.0, 0.0]])
rotx180 = rotx90 @ rotx90

roty90 = np.array([[0.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0],
                   [-1.0, 0.0, 0.0]])
roty180 = roty90 @ roty90

rotz90 = np.array([[0.0, -1.0, 1.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0]])
rotz180 = rotz90 @ rotz90

rot_world_to_kinect_ref = np.array([[-1.0, 0.0, 0.0],
                                    [0.0,  0.0, 1.0],
                                    [0.0,  1.0, 0.0]])
rot_kinect_to_world_ref = rot_world_to_kinect_ref.T


def align_data_by_timestamp(list_data_ids, list_timestamps,
                            undersample_rate=1, frames_clip_idx=(5, 5),
                            visualize=False, plot_data_names=None):
    """
    Aligns frames in directory by timestamp.

    Args:
        list_data_ids(Iterable): Iterable with ids of data to align.
        list_timestamps(Iterable): Iterable with timestamps to be used
            to align data.
        undersample_rate(int): Undersample rate to apply to data.
        frames_clip_idx(tuple[int]): Number of frames to ignore at the
            beginning/end.
        visualize(bool): If alignment results should be plotted.
        plot_data_names(None, Iterable[str]): Data type name to put
            in the plot.

    Returns:
        One list with frame ids for each of the dirs provided. Matching
            indexes of each list are the closest temporally.
    """

    # make timesteps relative to reference start
    min_len_size = np.inf
    init_t = list_timestamps[0][0]
    time_ref = (list_timestamps[0] - init_t)[frames_clip_idx[0]:-frames_clip_idx[1]:undersample_rate]
    frame_ref = list_data_ids[0][frames_clip_idx[0]:-frames_clip_idx[1]:undersample_rate]

    aligned_data_ids = [frame_ref, ]
    aligned_timestamps = [time_ref, ]
    for ts, d_id in zip(list_timestamps[1:], list_data_ids[1:]):
        ts, d_id = np.array(ts), np.array(d_id)

        rel_ts = (ts - init_t)
        # order samples by timestamp(allowing (0.5/30)ms ahead of ref
        align_ids = np.searchsorted(rel_ts, time_ref - (0.5/30))
        align_ids = (align_ids[align_ids != np.sort(align_ids)[-1]])     # remove left over not aligned samples
        aligned_timestamps.append(rel_ts[align_ids])
        min_len_size = min(len(align_ids), min_len_size)

        # select frames which align
        aligned_data_ids.append(d_id[align_ids])

    # guarantee all data has same lengths
    aligned_data_ids = [i[:min_len_size] for i in aligned_data_ids]
    aligned_timestamps = [i[:min_len_size] for i in aligned_timestamps]

    if visualize:
        if plot_data_names is None:
            plot_data_names = ["" for _ in range(len(list_data_ids))]

        # visualize aligned results
        fig, ax = plt.subplots(1, 2, figsize=(20*2, 20))
        plt.ylabel("timestamp", fontsize=40)
        # original timestamps
        for (tstamps, label) in zip(list_timestamps, plot_data_names):
            ts = (tstamps - init_t)[:min(25, min_len_size)*undersample_rate:undersample_rate]
            ax[0].plot(ts)
            ax[0].set_title("Original relative timestamps", fontsize=40)
            ax[0].set_xlabel("nframe", fontsize=40)
        # aligned timestamps
        for (tstamps, label) in zip(aligned_timestamps, plot_data_names):
            ts = tstamps[:min(25, min_len_size)]
            ax[1].plot(ts)
            ax[1].set_title("Aligned timestamps", fontsize=40)
            ax[1].set_xlabel("nframe", fontsize=40)
        plt.legend(plot_data_names, fontsize=40)
        plt.show()

    return aligned_data_ids, aligned_timestamps


def extract_timestamp_from_string(time_string, split_char="_"):
    '''
    Extracts formatted timestamp(s and ms from a string and returns its
    float value.
        ex. 0000000000_000000000 -> 0.0
    '''
    sec, ms = time_string.replace(", ", ".").split(split_char)
    sec = '{0:010d}'.format(int(sec))
    ms = '{0:09d}'.format(int(ms))
    return float('.'.join([sec, ms]))

def process_depth_frame(frame_depth, save_shape=None):
    # convert from mm(uint16) to meters(float32)
    frame_depth = frame_depth.astype(np.float32) / 1000.0

    if save_shape is not None:
        frame_depth = cv2.resize(frame_depth, dsize=save_shape,
                                 interpolation=cv2.INTER_NEAREST)
    return frame_depth[..., np.newaxis]


def draw_img_keypoints(img, keypoints, radius=1, color=(255, 255, 255, 255)):
    """
    Draws a list of keypoints into an image.
    """
    #print(img.shape, img.max(), img.dtype)
    #new_img=img.copy()
    new_img = (cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
               if img.shape[-1] == 1 else img.copy())
    for point in keypoints:
        new_img = cv2.circle(new_img, tuple(point),
                             radius=radius, color=color,
                             thickness=-1)
    return new_img


def apply_homogeneous_transform(points3d, homog_transform):
    """
    Apply an homogeneous transform matrix to an array of 3D points.

    Args:
        points3d(np.array[Nx3]): 3d coordinate points.
        homog_transform(np.array[4x4]): homogeneous transformation matrix

    Returns:
        transformed(np.array[Nx3]) 3d new coordinate points.
    """

    homog_points3d = np.append(points3d, np.ones((points3d.shape[0], 1)), axis=1)
    transformed = (homog_transform @ homog_points3d.T).T
    return transformed[:, :3]


def project_3d_to_2d(points3d, intrinsic_matrix, extrinsic_matrix=None):
    """
    Project points from 3d to 2d given camera intrinsic and extrinsic
        matrices:
        # https://ksimek.github.io/2013/08/13/intrinsic/
        # https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf

    intrinsic_matrix =
                [[fx,  s,   x0,  0],
                 [0,   fy,  y0,  0],
                 [0,   0,   1,   0],
                 [0,   0,   0,   1]]

    extrinsic_matrix =
                [[Rxx, Rxy, Rxz, tx],
                 [Ryx, Ryy, Ryz, ty],
                 [Rzx, Rzy, Rzz, tz],
                 [0,   0,   0,   1]]

    Args:
        points3d (Nx3 array): points in world coordinates
        intrinsic_matrix (4x4 array): Camera intrinsic matrix.
        extrinsic_matrix (None, 4x4 array): Camera extrinsic matrix.
            If None then it is assumed that the 3D referential is in
            relation to the camera.

    Returns:
        points2d(Nx2 array): Projected points in pixel space.
    """

    # projection matrix (intrinsic * extrinsic) or just (intrinsic) if no
    # extrinsic transformation needed
    proj_matrix = (intrinsic_matrix @ extrinsic_matrix
                   if extrinsic_matrix is not None
                   else intrinsic_matrix)

    homog_points2d = apply_homogeneous_transform(points3d, proj_matrix)

    # ((x, y, z) -> (fx*x/z, fy*y/z))
    # no need to rescale as focal intrinsic params are in pixel space and not in mm.
    fx = 1  #intrinsic_matrix[0, 0]
    fy = 1  #intrinsic_matrix[1, 1]
    points2d = np.array([fx * homog_points2d[:, 0] / homog_points2d[:, 2],
                         fy * homog_points2d[:, 1] / homog_points2d[:, 2]]).T
    return points2d


def depth_to_pointcloud(depth_frame, camera_intrinsics, img_frame=None,
                        depth_range=(0.5, 5.0), num_points_pc=2048,
                        downsample_method=None, depth_mult=1.0):
    """
    Convert depth map to 3D pointcloud.

    Args:
        depth_frame(np.ndarray): depth image frame to project to 3D space.
        camera_intrinsics(np.ndarray): camera intrinsic parameters. Should
            have [4x4] shape. Check project_3d_to_2d function for details.
        img_frame(np.ndarray, None): Optional color for pointcloud. Should
            have the same shape as "depth_frame", range [0.0, 1.0] and
            (H,W,C) order.
        depth_range(tuple[float]): valid depth range. Values outside
            this range are ignored.
        num_points_pc(int): number of points to have for the pointcloud.
            If num_points_pc < depth_pixels an undersample method is
            applied.
        downsample_method(None, str): Undersample method to apply to
            pointcloud data. Can be one of "linspace", "random" or None.
            If None, then no undersample is applied.
        depth_mult(float): multiplier to apply to depth data.

    Returns:
         pointcloud(np.ndarray): pointcloud datapoints.
         pc_color(np.ndarray): color of each of the pointcloud points.
    """

    # apply corrections to depth data:
    # - convert from [0-1] range to kinect original uint32 range(in mm)
    # - convert from kinect mm values to m
    depth_frame = np.squeeze(depth_frame).astype(np.float32)
    depth_frame = depth_frame * depth_mult

    # drop depth data outside valid range
    valid_depth = (depth_range[0] < depth_frame) & (depth_frame < depth_range[1])
    img_y, img_x = np.argwhere(valid_depth).T
    z = depth_frame[np.where(valid_depth)]

    # downsample pointcloud
    if downsample_method is None or num_points_pc >= len(z):
        pass
    elif downsample_method == "linspace":
        selec_points = np.linspace(0, len(z), num=num_points_pc, endpoint=False, dtype=np.int32)
        img_x, img_y, z = img_x[selec_points], img_y[selec_points], z[selec_points]
    elif downsample_method == "random":
        selec_points = np.random.randint(0, len(z), size=num_points_pc)
        img_x, img_y, z = img_x[selec_points], img_y[selec_points], z[selec_points]
    else:
        raise NotImplementedError

    # (xy_pos - cam_center_xy) * z / cam_focalLength
    # invert (-x / -y) to transform pixel ref to world ref
    x = -((img_x - camera_intrinsics[0, 2]) * z) / camera_intrinsics[0, 0]
    y = -((img_y - camera_intrinsics[1, 2]) * z) / camera_intrinsics[1, 1]
    pointcloud = np.stack([x, y, z]).T

    # get pointcloud color from image_frame
    if img_frame is not None:
        pc_color = img_frame[img_y.astype(np.int32), img_x.astype(np.int32)]
    else:
        pc_color = (0.5, 0.5, 0.5, 0.5)

    return pointcloud, pc_color


def multiple_cams_merge_pointcloud(depth_data, intrinsic_matrix,
                                   extrinsic_matrix=None, n_points=4096,
                                   depth_range=(0.1, 2.5)):
    """
    Creates pointcloud from multiple depth cameras and joins them to
    create a single pointcloud.

    Args:
        depth_data(Iterable(np.ndarray): Depth data from cameras to merge.
        intrinsic_matrix(Iterable(np.ndarray): Intrinsics parameters from
            cameras to merge.
        extrinsic_matrix(Iterable(np.ndarray): Extrinsic transformation to
            apply to each of the cameras to merge.
        n_points(int): Number of desired points for resulting pointcloud.
        depth_range(tuple[float]): range of depth values to keep.
    """

    extrinsic_matrix = (extrinsic_matrix if extrinsic_matrix is not None
                        else [np.eye(4) for c in range(len(intrinsic_matrix))])

    cam_pc, cam_pccolor = [], []
    for c_dpt, c_intrinsics, c_extrinsics in \
            zip(depth_data, intrinsic_matrix, extrinsic_matrix):
        c_pc, c_clr = depth_to_pointcloud(
                depth_frame=c_dpt,
                camera_intrinsics=c_intrinsics,
                img_frame=None,
                depth_range=depth_range, num_points_pc=n_points,
                downsample_method="random")

        cam_pc.append(apply_homogeneous_transform(c_pc, c_extrinsics))
        cam_pccolor.append(c_clr)

    return np.concatenate(cam_pc), np.concatenate(cam_pccolor)


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Finds the rotation matrix that aligns vec1 to vec2.

    Args:
        vec1(np.ndarray): A 3d "source" vector
        vec2(np.ndarray): A 3d "destination" vector

    Returns:
        rotation matrix(np.array[3x3]) which when applied to vec1, aligns
        it with vec2.
    """
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
