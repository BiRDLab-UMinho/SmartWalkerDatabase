
import glob
import json
import os
import re
import time
import c3d

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as scipyR

from utils import project_3d_to_2d,align_data_by_timestamp,rot_world_to_kinect_ref,\
    roty90, process_depth_frame, draw_img_keypoints, extract_timestamp_from_string, \
    frames_original_shape, xsens_joint_names


def get_skeleton_data_from_xsens(data_path, extrinsic_ref_transforms=None, load_c3d=True,
                                 skeleton_norm_mode=("norm_pos_orient", "camera_ref")):

    """
    Parses data from exported Xsens files(.csv / .c3d) and aligns data
    spatially to desired referential.

    Args:
        data_path(str): path to Xsens data.
        extrinsic_ref_transforms(dict): dictionary with extrinsic
            calibration data.
        load_c3d(bool): if joint data in .c3d files should be used. The
            .c3d files contains a more complete skeleton joint set and in
            this case is used to replace the joints in the feet.
        skeleton_norm_mode(tuple[str]): mode to use when aligning the
            skeleton data. Can use more than one mode to save data in different
            ways:
                -"camera_ref": aligns skeleton to be as seen from the
                    posture camera referential. This is the default mode
                    and enables projection to 2D frames.
                -"norm_pos_orient": normalizes skeleton orientation and
                    position, skeleton is centered on root joint and always
                     faces forward. Might be the best option when dealing
                     with only 3D.
                -"none": uses raw skeleton data. Skeleton moves and rotates in
                    space.

    Returns:
        xsense_frame_idx, qpos3d_data processed with selected methods

    """

    # load joint data(3D joint positions, 3D root position and angles) from Xsens files
    qpos3d_data = pd.read_csv(data_path + "_csv/Segment Position.csv",
                              sep=";", index_col=0).values
    qpos3d_com = pd.read_csv(data_path + "_csv/Center of Mass.csv",
                             sep=";", index_col=0).values
    qangle_euler_xyz = pd.read_csv(data_path + "_csv/Segment Orientation - Euler.csv",
                                   sep=";", index_col=0).values

    # reshapes data from [Nsamples, Joint * 3]  to [Nsamples, Joint, 3]
    qpos3d_data = qpos3d_data.reshape(len(qpos3d_data), -1, 3)  # reshape to [n_samples, n_joints, pos_xyz]
    qpos3d_com = qpos3d_com.reshape(len(qpos3d_com), 1, 3)      # reshape to [n_samples, 1, pos_xyz]
    qangle = np.deg2rad(qangle_euler_xyz.reshape(len(qangle_euler_xyz), -1, 3))  # reshape to [n_samples, n_joints, euler_angle_xyz] and convert to rad

    # extract necessary data when normalizing data to kinect referential
    if "camera_ref" in skeleton_norm_mode:
        assert qpos3d_data.shape[1] == 24, "Prop sensor data is necessary to align skeleton" \
                                           "to camera_ref!"
        assert extrinsic_ref_transforms is not None, "Extrinsic transformation data is " \
                                                     "necessary to align skeleton to camera_ref"

        # separate data from prop sensor
        qpos3d_data = qpos3d_data[:, :23, :]
        prop_angle = qangle[:, 23, :]
        qangle = qangle[:, :23, :]

        # offset to walker handles in world ref from extrinsic calibration files
        # (points know to be relatively fixed as people need to grab them - walker handles)
        walker_offset_pos = dict(
                left=extrinsic_ref_transforms["CamPostureToLeftHandleTranslation"],
                right=extrinsic_ref_transforms["CamPostureToRightHandleTranslation"])

        # reads optional rotation from extrinsic calibration file
        # (this was added to correct bad placement of the prop sensor on
        # the camera in some tests - ex. sensor placed facing opposite direction)
        xsens_camera_external_rot = extrinsic_ref_transforms["PropOrientation"]

        # if skeleton z_orientation should be relative to the prop sensor placed on camera
        # (its off by default as some trials exhibit drift from the prop sensor resulting
        # in incorrect rotation relative to the camera. In most cases people walk facing the
        # camera so orientation in the z-axis is close to 0, however this needs to be turned
        # on when people dont face the camera when walking(a flag is set in the external
        # calibration files in theses cases).
        if ("UsePropDirection" in extrinsic_ref_transforms
                and any([(f in data_path) for f in extrinsic_ref_transforms["UsePropDirection"]])):
            use_direction_from_prop = True
        else:
            use_direction_from_prop = False

    else:
        # separate data from prop sensor
        qpos3d_data = qpos3d_data[:, :23, :]
        prop_angle = None
        qangle = qangle[:, :23, :]
        walker_offset_pos = None
        xsens_camera_external_rot = None
        use_direction_from_prop = False

    if load_c3d:
        # read c3d data from file
        with open(data_path + ".c3d", 'rb') as fhandle:
            c3d_data = []
            for frame_no, points, analog in c3d.Reader(fhandle).read_frames(copy=False):
                # pts_data = points[np.ix_(c3d_extra_points_selec, [0, 1, 2])] / 1000.0
                pts_data = points[:, [0, 1, 2]] / 1000.0
                c3d_data.append(pts_data)

        # check if Xsens data was exported correctly or apply small fix to correct
        if len(qpos3d_data) != len(c3d_data):
            #print("Warning: len(qpos3d_data) != len(c3d_qpos3d_data) -> {}|{}.  "
            #      "This seems due to Xsens exporting bugs. Applying small fix to correct!"
            #      .format(qpos3d_data.shape, np.shape(c3d_data)))
            qpos3d_data = qpos3d_data[:len(c3d_data)]
            qpos3d_com = qpos3d_com[:len(c3d_data)]
            qangle = qangle[:len(c3d_data)]
            prop_angle = prop_angle[:len(c3d_data)]

        # replace keypoints of both feet(toe/heel)
        c3d_qpos3d_data = np.array(c3d_data)
        qpos3d_data[:, [17, 18, 21, 22]] = c3d_qpos3d_data[:, [56, 55, 62, 61]]

    # aligns skeleton data
    qpos3d_processed_data = normalize_skeleton(
            pos3d=qpos3d_data, qangle=qangle,
            com_pos=qpos3d_com, prop_angle=prop_angle,
            walker_offset_pos_lr_handles=walker_offset_pos,
            skeleton_external_rot=xsens_camera_external_rot,
            skeleton_norm_mode=skeleton_norm_mode,
            use_prop_direction=use_direction_from_prop)

    xsense_frame_idx = np.arange(len(qpos3d_data))

    return xsense_frame_idx, qpos3d_processed_data


def normalize_skeleton(pos3d, qangle, com_pos,
                       skeleton_norm_mode=("norm_pos_orient", "camera_ref"),
                       walker_offset_pos_lr_handles=None, prop_angle=None,
                       use_prop_direction=False, skeleton_external_rot=None,
                       root_idx=0, r_wrist_idx=10, l_wrist_idx=14):
    """
    Normalizes skeleton data using the desired modes:

    Args:
        pos3d(np.ndarray): array with 3D joint data positions. Shape:[N, J, 3]
        qangle(np.ndarray): array with segment angles. Shape:[N, J, 3]
        com_pos(np.ndarray): array with center-of-mass positions. Shape:[N, 1, 3]
        skeleton_norm_mode(tuple[str]): mode to use when aligning the
            skeleton data. Can use more than one mode to process data in
            different ways:
                -"camera_ref": aligns skeleton to be as seen from the
                    posture camera referential. This is the default mode
                    and enables projection to 2D frames.
                -"norm_pos_orient": normalizes skeleton orientation and
                    position, skeleton is centered on root joint and always
                     faces forward. Might be the best option when dealing
                     with only 3D.
                -"none": uses raw skeleton data. Skeleton moves and rotates in
                    space.
        walker_offset_pos_lr_handles(None, dict[np.ndarray]): dictionary with
            translation vector from posture camera to each of the walker
            handles("left"/"right"). Only necessary when using "camera_ref" mode.
        prop_angle(None, np.ndarray): array with prop angles.
            Shape:[N, 1, 3]. Only necessary when using "camera_ref" mode.
        use_prop_direction(bool): if prop rotation should be used to normalize
            skeleton z_orientation relative to camera.
        skeleton_external_rot(None, np.ndarray): optional rotation to fix bad
            prop sensor placement in some trials.
        root_idx(int): Idx of the root joint.
        r_wrist_idx(int): Idx of the right wrist joint.
        l_wrist_idx(int): Idx of the left wrist joint.

    Returns:
        list with each set of the joint positions processed with selected methods
    """
    processed_skeletons = []

    # center root joint on origin before doing further transforms
    root_pos = pos3d.copy()[:, root_idx, :].reshape(len(pos3d), 1, 3)
    pos3d_orig = pos3d - root_pos

    if "camera_ref" in skeleton_norm_mode:
        pos3d_cam = pos3d_orig.copy()
        # rotate skeleton referential to kinect camera referential
        # (prop_idx)camera ref is prone to rotation drift in some cases
        # (root_idx)(removing root_z_rotation to deal with this - not as correct)
        if use_prop_direction:
            xsens_prop_to_world_ref = scipyR.from_rotvec(
                    (prop_angle) * [0, 0, 1]).as_matrix()
        else:
            xsens_prop_to_world_ref = scipyR.from_rotvec(
                    (qangle[:, root_idx, :]) * [0, 0, 1]).as_matrix()
        pos3d_cam = (pos3d_cam @ xsens_prop_to_world_ref) @ (rot_world_to_kinect_ref @ roty90)

        # rotate skeleton to match kinect camera tilt angle
        camera_tilt = scipyR.from_rotvec(
                (prop_angle + [np.pi / 2, 0, 0]) * [1, 1, 0]).as_matrix()
        pos3d_cam = (pos3d_cam @ camera_tilt)

        # apply external skeleton rotation if needed(sometimes to fix some acquisition problems)
        external_skeleton_rot = scipyR.from_rotvec(skeleton_external_rot).as_matrix()
        pos3d_cam = pos3d_cam @ external_skeleton_rot

        # transl from camera to right wrist is known as well as from r_wrist to root.
        l_hand_to_root_offset = (pos3d_cam[:, root_idx, :].reshape(len(pos3d_cam), 1, 3)
                                 - pos3d_cam[:, l_wrist_idx, :].reshape(len(pos3d_cam), 1, 3))
        r_hand_to_root_offset = (pos3d_cam[:, root_idx, :].reshape(len(pos3d_cam), 1, 3)
                                 - pos3d_cam[:, r_wrist_idx, :].reshape(len(pos3d_cam), 1, 3))

        # average transformation from
        pos3d_l = pos3d_cam + walker_offset_pos_lr_handles["left"] + l_hand_to_root_offset
        pos3d_r = pos3d_cam + walker_offset_pos_lr_handles["right"] + r_hand_to_root_offset
        pos3d_cam = (pos3d_l + pos3d_r) / 2

        # fix bad wrist positions
        pos3d_cam = _fix_xsens_wrist_postions(pos3d_cam, walker_offset_pos_lr_handles,
                                              threshold_m=35e-3)
        processed_skeletons.append(pos3d_cam)

    # convert to center of mass centered referential
    if "norm_pos_orient" in skeleton_norm_mode:
        pos3d_norm = pos3d_orig.copy()

        # normalize joint orientation over z-axis
        qangle[:, :, [0, 1]] = 0.0
        root_rot = scipyR.from_rotvec(qangle[:, root_idx, :]).as_matrix()
        pos3d_norm = pos3d_norm @ root_rot

        # normalize joint positions(center model center_of_mass
        # on the x/y/z axis for all timesteps)
        pos3d_norm = pos3d_norm + root_pos
        pos3d_norm = pos3d_norm - com_pos
        processed_skeletons.append(pos3d_norm)

    # just use raw data
    if "none" in skeleton_norm_mode:
        pos3d_raw = pos3d_orig + root_pos
        processed_skeletons.append(pos3d_raw)

    return processed_skeletons


def _fix_xsens_wrist_postions(qpos3d, walker_offset_pos_lr_handles, threshold_m=30e-3,
                              r_wrist_idx=10, l_wrist_idx=14):
    # replace average(still maintains deviations) wrist joints with
    # default wrist positions(joint handles) if wrist positions are
    # far far from handles(ex. because of bad xsens calibration)
    fixed_qpos3d = qpos3d.copy()
    r_wrist_data_mean = qpos3d[:, r_wrist_idx, :].mean(axis=0)
    l_wrist_data_mean = qpos3d[:, l_wrist_idx, :].mean(axis=0)

    r_wrist_offset = np.linalg.norm(r_wrist_data_mean - walker_offset_pos_lr_handles["right"])
    l_wrist_offset = np.linalg.norm(l_wrist_data_mean - walker_offset_pos_lr_handles["left"])

    # replace right wrist mean
    if r_wrist_offset > threshold_m:
        fixed_qpos3d[:, r_wrist_idx, :] = ((qpos3d[:, r_wrist_idx, :] - r_wrist_data_mean)
                                           + walker_offset_pos_lr_handles["right"])
    # replace left wrist mean
    if l_wrist_offset > threshold_m:
        fixed_qpos3d[:, l_wrist_idx, :] = ((qpos3d[:, l_wrist_idx, :] - l_wrist_data_mean)
                                           + walker_offset_pos_lr_handles["left"])
    return fixed_qpos3d


def cams_project_3d_to_2d(points3d, intrinsic_matrix, extrinsic_matrix=None,
                          mode="separate", resize_factor=(1.0, 1.0),
                          cam2_pixel_offset=(0, 480)):
    """
    Projects 3D points to cameras' frame(2D).

    Args:
        points3d(np.ndarray): Points to project from 3D to 2D space.
        intrinsic_matrix(dict[np.ndarray]): Dict with intrinsic matrix
            for each camera for the projection. Should have shape:[4x4].
        extrinsic_matrix(dict[np.ndarray]): Dict with extrinsic matrix
            for each camera for the projection. Should have shape:[4x4].
        mode(str): Projection mode for multiple cameras:
                "separate" - returns 2D points relative to each camera,
                    frame in separate.
                "concatenated" - assumes the gait camera frame is
                    concatenated bellow the posture camera frame, creating
                    a single frame.
        resize_factor(tuple[float]): Resize factor(x, y to apply to the
            projected points if the frame shape is altered.
        cam2_pixel_offset(tuple[int]): position offset for second camera
            frame when mode is "concatenated".

    Returns:
         Points projected in 2D image space
    """

    extrinsic_matrix = (extrinsic_matrix if extrinsic_matrix is not None
                        else {c: np.eye(4) for c in intrinsic_matrix.keys()})

    cam_qpos2d_posture = project_3d_to_2d(points3d=points3d,
                                          intrinsic_matrix=intrinsic_matrix["posture_camera"],
                                          extrinsic_matrix=extrinsic_matrix["posture_camera"])

    cam_qpos2d_gait = project_3d_to_2d(points3d=points3d,
                                       intrinsic_matrix=intrinsic_matrix["gait_camera"],
                                       extrinsic_matrix=extrinsic_matrix["gait_camera"])

    # determines in which image part is the joint located(and choose that projection position)
    qpos_posture_off = cam_qpos2d_posture[:, 1] - cam2_pixel_offset[1]
    qpos_gait_off = -cam_qpos2d_gait[:, 1]

    if mode == "separate":
        cam_qpos2d_gait = cam_qpos2d_gait * list(resize_factor)
        cam_qpos2d_posture = cam_qpos2d_posture * list(resize_factor)
        return cam_qpos2d_gait, cam_qpos2d_posture

    elif mode == "concatenated":
        cams_qpos2d = np.where((qpos_posture_off < qpos_gait_off)[:, np.newaxis],
                               cam_qpos2d_posture,
                               cam_qpos2d_gait + cam2_pixel_offset)
        cams_qpos2d = cams_qpos2d * [resize_factor[0], resize_factor[1]/2]
        return cams_qpos2d

    else:
        raise NotImplementedError


def parse_walker_data(
        dataset_root_path, subj_ids_to_extract, seq_ids_to_extract, rep_ids_to_extract,
        save_path=None, save_data=True, show_data=False, undersample_rate=1, ignore_error=True):
    """
    Function to parse raw walker data:
        -Aligns data temporally from gait/posture cameras and Xsens skeleton
        -Correlates Xsens 3D joint data spatially to posture camera referential.
        -Projects 3D skeleton to each of the cameras' frame.

        -Example of how to parse additional data (Depth/Pointcloud) which
            is not used by default as it takes longer to process and occupies
            significant space in disk.
        -Joint angles can also be extracted from xsens file, however are not
            being parsed by default(check "Joint Angles ..." tabs).
        -Optional visualization of 2D data(depth frames with projected
            2D joints).

    Args:
        dataset_root_path(str): path to root of the dataset.
        subj_ids_to_extract(Iterable[str]): selected subjects to extract data.
        seq_ids_to_extract(Iterable[str]):  selected sequences to extract data.
        rep_ids_to_extract(Iterable[str]): selected repetitions to extract data.
        save_path(str): path to save extracted data.
        save_data(bool): if data should be saved to disc.
        show_data(bool): if data should be visualized(projected 2D keypoints
            on Depth frames).
        undersample_rate(int): undersample rate to apply to data. 1 means
            no undersample.
        ignore_error(bool): if errors should be ignored while parsing.

    """

    if save_path is None:
        save_path = dataset_root_path + "/processed_data"
    os.makedirs(save_path, exist_ok=True)  # make sure dir exists

    # go to raw trial directory
    dataset_root_path = dataset_root_path + "/raw_data"

    print("Extracting dataset data, this will take a while!")
    n_frames = 0
    for subj_id in sorted(subj_ids_to_extract):
        for seq_id in sorted(seq_ids_to_extract):
            for rep_id in sorted(rep_ids_to_extract):

                seq_path_name = subj_id + "_" + seq_id
                rep_path_name = seq_path_name + "_" + rep_id

                data_tree_path = "/{}/{}/{}/" \
                    .format(subj_id, seq_path_name, rep_path_name)

                if not os.path.isdir(dataset_root_path + data_tree_path):
                    # ignores data combinations which dont exist or are not available
                    print("Requested data does not exist: {} | {} | {}"
                          .format(subj_id, seq_id, rep_id))
                    continue

                extrinsic_calib_path = dataset_root_path + "/{}/{}_extrinsic_calibration.json"\
                    .format(subj_id, subj_id)
                intrinsic_calib_path = dataset_root_path + "/{}/{}_intrinsic_calibration.json"\
                    .format(subj_id, subj_id)

                try:
                    time_start = time.perf_counter()

                    # load extrinsic referential transforms + intrinsic camera params
                    with open(extrinsic_calib_path, "r") as f:
                        extrinsic_ref_transforms = json.load(f)

                    with open(intrinsic_calib_path, "r") as f:
                        cam_intrinsics = json.load(f)
                        for k in cam_intrinsics:
                            cam_intrinsics[k] = np.array(cam_intrinsics[k])

                    # get path to files
                    skeleton_data_file_path = dataset_root_path + data_tree_path + "{}"\
                        .format(rep_path_name)
                    depth_g_path = dataset_root_path + data_tree_path + "gait_depth_registered"
                    depth_p_path = dataset_root_path + data_tree_path + "posture_depth_registered"

                    ##### get depth data indexes and timestamps #####
                    depth_fnum_pattern = re.compile("[a-zA-Z]+_[a-zA-Z]+_(\d+)_\d+_\d+.png")
                    depth_g_frame_ids = sorted(
                            os.listdir(depth_g_path),
                            key=lambda s: int(re.search(depth_fnum_pattern, s).group(1)))
                    depth_p_frame_ids = sorted(
                            os.listdir(depth_p_path),
                            key=lambda s: int(re.search(depth_fnum_pattern, s).group(1)))

                    depth_time_pattern = re.compile("[a-zA-Z]+_[a-zA-Z]+_\d+_(\d+_\d+).png")
                    depth_extract_tstep_func = lambda s: extract_timestamp_from_string(
                            re.findall(depth_time_pattern, s)[0], split_char="_")
                    depth_g_timestamps = np.array([depth_extract_tstep_func(f_id)
                                                   for f_id in depth_g_frame_ids])
                    depth_p_timestamps = np.array([depth_extract_tstep_func(f_id)
                                                   for f_id in depth_p_frame_ids])

                    depth_g_frame_idx = [int(re.search(depth_fnum_pattern, s).group(1))
                                         for s in depth_g_frame_ids]
                    depth_p_frame_idx = [int(re.search(depth_fnum_pattern, s).group(1))
                                         for s in depth_p_frame_ids]

                    ##### get xsens data indexes and timestamps #####

                    # get skeleton joints3D data processed by the desired methods
                    # and corresponding indexes. The "camera_ref" presents the data relative
                    # to the posture camera while the "norm_pos_orient" has centered root
                    # position and oriented facing forward.
                    xsens_frame_idx, (qpos3d_camref_data, qpos3d_norm_data) = \
                        get_skeleton_data_from_xsens(
                                data_path=skeleton_data_file_path,
                                extrinsic_ref_transforms=extrinsic_ref_transforms,
                                skeleton_norm_mode=("camera_ref", "norm_pos_orient"),
                                load_c3d=True)

                    xsens_start_time = [float(os.path.basename(f).replace("sync_", "")
                                              .replace(".stamp", "").replace("_", "."))
                                        for f in glob.glob(dataset_root_path + data_tree_path
                                                           + "/sync_*.stamp")]
                    xsens_start_time = (xsens_start_time[0] if xsens_start_time
                                        else depth_g_timestamps[0] - 0.65)
                    xsens_timestamps = np.array((xsens_frame_idx * (1/60)) + xsens_start_time)

                    ##### align depth and xsense data temporally based on timestamps #####
                    (dpt_p_idx, dpt_g_idx, xsens_idx), _ = \
                        align_data_by_timestamp(
                            list_data_ids=[depth_p_frame_idx, depth_g_frame_idx,
                                           xsens_frame_idx],
                            list_timestamps=[depth_p_timestamps, depth_g_timestamps,
                                             xsens_timestamps],
                            frames_clip_idx=(10, 5), undersample_rate=undersample_rate,
                            plot_data_names=["depth_posture", "depth_gait",
                                             "xsense"], visualize=False)

                    # indexes of aligned data
                    os.makedirs(save_path + data_tree_path, exist_ok=True)  # make sure dir exists
                    fhandle_idx_align = pd.DataFrame(
                            columns=["depth_posture_idx", "depth_gait_idx",
                                     "xsense_idx"],
                            data={"depth_posture_idx": dpt_p_idx,
                                  "depth_gait_idx":    dpt_g_idx,
                                  "xsense_idx":        xsens_idx})
                    fhandle_idx_align.to_csv(save_path + data_tree_path
                                             + "synchronized_data_idx.csv")

                    # parse and save aligned data
                    out_qpos3d_norm_data, out_qpos3d_camref_data, \
                    out_qpos2d_gait_data, out_qpos2d_posture_data = [], [], [], []
                    num_frames_extracted = 0
                    for f_i, (f_dpt_p_idx, f_dpt_g_idx, f_xsens_idx) \
                            in enumerate(zip(dpt_p_idx, dpt_g_idx, xsens_idx)):

                        # select 3D data
                        f_qpos3d_norm = qpos3d_norm_data[f_xsens_idx]
                        f_qpos3d_camref = qpos3d_camref_data[f_xsens_idx]

                        # obtain 2D data from projection of skeleton relative to camera
                        f_qpos2d_gait, f_qpos2d_posture = cams_project_3d_to_2d(
                                points3d=f_qpos3d_camref * [-1, -1, 1],
                                intrinsic_matrix=cam_intrinsics,
                                extrinsic_matrix=dict(
                                        gait_camera=extrinsic_ref_transforms[
                                            "CamGaitToPostureTransform"],
                                        posture_camera=np.eye(4)),
                                mode="separate")

                        # save 3D and 2D data
                        out_qpos3d_norm_data.append(f_qpos3d_norm)
                        out_qpos3d_camref_data.append(f_qpos3d_camref)

                        out_qpos2d_gait_data.append(np.round(f_qpos2d_gait).astype(np.int32))
                        out_qpos2d_posture_data.append(np.round(f_qpos2d_posture).astype(np.int32))

                        num_frames_extracted += 1

                        if show_data:
                            # example of how to process/visualize frames with overlaid
                            # keypoint data.
                            # (not being used by default as it takes longer to process
                            # but can be used for visualization/debugging/additional processing
                            show_shape = (256, 224)
                            f_dpt_g = process_depth_frame(
                                    cv2.imread(
                                        depth_g_path + "/" + depth_g_frame_ids[f_dpt_g_idx],
                                        cv2.IMREAD_ANYDEPTH),
                                    save_shape=show_shape).astype(np.float32)

                            f_dpt_p = process_depth_frame(
                                    cv2.imread(
                                        depth_p_path + "/" + depth_p_frame_ids[f_dpt_p_idx],
                                        cv2.IMREAD_ANYDEPTH),
                                    save_shape=show_shape).astype(np.float32)

                            # resize keypoints 2D to match resized frames
                            f_qpos2d_data_gait_show = np.round(
                                    f_qpos2d_gait * np.divide(show_shape,
                                                              frames_original_shape)
                            ).astype(np.int32)

                            f_qpos2d_data_posture_show = np.round(
                                    f_qpos2d_posture * np.divide(show_shape,
                                                                 frames_original_shape)
                            ).astype(np.int32)

                            # draw keypoints on depth frames
                            depth_gait_frame_show = draw_img_keypoints(
                                    f_dpt_g * 0.1, f_qpos2d_data_gait_show,
                                    color=(0, 1, 0, 1), radius=2)
                            depth_posture_frame_show = draw_img_keypoints(
                                    f_dpt_p * 0.1, f_qpos2d_data_posture_show,
                                    color=(0, 1, 0, 1), radius=2)

                            # show data
                            cv2.imshow("Depth gait frame", depth_gait_frame_show)
                            cv2.imshow("Depth posture frame", depth_posture_frame_show)
                            cv2.waitKey(1)

                    assert len(out_qpos3d_norm_data) == len(out_qpos3d_camref_data) == \
                           len(out_qpos2d_gait_data) == len(out_qpos2d_posture_data) == \
                           num_frames_extracted, \
                        "Not all data has the same lenght! Check your files: " \
                        "{} | {} | {},  Lens: {} | {} | {} | {} | {}".format(
                                subj_id, seq_id, rep_id,
                                len(out_qpos3d_norm_data), len(out_qpos3d_camref_data),
                                len(out_qpos2d_gait_data), len(out_qpos2d_posture_data),
                                num_frames_extracted)

                    if save_data:
                        # save 3d aligned data to original dataset
                        out_qpos3d_norm_data = np.round(np.array(out_qpos3d_norm_data), 4)
                        out_qpos3d_camref_data = np.round(np.array(out_qpos3d_camref_data), 4)
                        out_qpos2d_gait_data = np.array(out_qpos2d_gait_data)
                        out_qpos2d_posture_data = np.array(out_qpos2d_posture_data)

                        jnames3d, jnames2d = [], []
                        for n in xsens_joint_names:
                            jnames3d.extend([n + "_x", n + "_y", n + "_z"])
                            jnames2d.extend([n + "_x", n + "_y"])

                        # save 3D skeleton data normalized
                        fhandle_idx_align = pd.DataFrame(
                                columns=jnames3d,
                                data=out_qpos3d_norm_data.reshape(
                                        (len(out_qpos3d_norm_data), -1)))
                        fhandle_idx_align.to_csv(save_path + data_tree_path
                                                 + "normalized_skeleton_3d.csv")

                        # save 3D skeleton data aligned with posture camera referential
                        fhandle_idx_align = pd.DataFrame(
                                columns=jnames3d,
                                data=out_qpos3d_camref_data.reshape(
                                        (len(out_qpos3d_camref_data), -1)))
                        fhandle_idx_align.to_csv(save_path + data_tree_path
                                                 + "aligned_skeleton_3d.csv")

                        # save 2D skeleton data projected to gait camera frames
                        fhandle_idx_align = pd.DataFrame(
                                columns=jnames2d,
                                data=out_qpos2d_gait_data.reshape(
                                        (len(out_qpos2d_gait_data), -1)))
                        fhandle_idx_align.to_csv(save_path + data_tree_path
                                                 + "aligned_skeleton_2d_gait.csv")

                        # save 2D skeleton data projected to posture camera frames
                        fhandle_idx_align = pd.DataFrame(
                                columns=jnames2d,
                                data=out_qpos2d_posture_data.reshape(
                                        (len(out_qpos2d_posture_data), -1)))
                        fhandle_idx_align.to_csv(save_path + data_tree_path
                                                 + "aligned_skeleton_2d_posture.csv")

                    n_frames += num_frames_extracted
                    print("Extracted data from: {} | {} | {}   ->   Samples: {} | {}s"
                          .format(subj_id, seq_id, rep_id, num_frames_extracted,
                                  round(time.perf_counter() - time_start, 2)))

                except Exception as e:
                    if not ignore_error:
                        raise e
                    else:
                        print("Could not extract data from: {} | {} | {}"
                              .format(subj_id, seq_id, rep_id),
                              " --------------------------------------->  Error: ", e)

    if show_data:
        cv2.destroyAllWindows()

    print("Extracted a total of {} data samples".format(n_frames))


if __name__ == "__main__":
    dataset_path = "../.."

    # subject data to extract
    subj_ids = ["participant{0:02d}".format(i) for i in range(14)]

    # sequence data to extract for each subject
    seq_ids = ["left_0.3", "left_0.5", "left_0.7",
               "right_0.3", "right_0.5", "right_0.7",
               "straight_0.3", "straight_0.5", "straight_0.7"]

    rep_ids = ["corner1", "corner2", "corner3",
               "corridor1", "corridor2", "corridor3"]

    parse_walker_data(
            dataset_root_path=dataset_path,

            subj_ids_to_extract=subj_ids,
            seq_ids_to_extract=seq_ids,
            rep_ids_to_extract=rep_ids,

            save_path="./parsing_output/",
            save_data=True,
            show_data=True,
            ignore_error=True,
            undersample_rate=1)
