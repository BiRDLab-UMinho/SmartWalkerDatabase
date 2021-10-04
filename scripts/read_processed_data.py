
import json
import os
import re

import cv2
import numpy as np
import pandas as pd

from utils import process_depth_frame, xsens_joint_names, \
    frames_original_shape, multiple_cams_merge_pointcloud, \
    draw_img_keypoints


def read_walker_data(
        dataset_root_path, subj_ids, seq_ids, rep_ids,
        frames_shape=(320, 240), undersample_rate=1,
        ignore_error=True):
    """
    Function to read/use walker data.

    Args:
        dataset_root_path(str): path to root of the walker dataset.
        subj_ids(Iterable[str]): selected subjects read data.
        seq_ids(Iterable[str]):  selected sequences read data.
        rep_ids(Iterable[str]): selected repetitions read data.
        frames_shape(tuple[int]): target size for camera frames.
        undersample_rate(int): Undersample rate to apply to data. 1 means
            no undersample.
        ignore_error(bool): if errors should be ignored while parsing.
    """

    dataset_raw_path = dataset_root_path + "/raw_data"
    dataset_proc_path = dataset_root_path + "/processed_data"

    print("Extracting dataset data, this will take a while!")
    for subj_id in sorted(subj_ids):
        for seq_id in sorted(seq_ids):
            for rep_id in sorted(rep_ids):

                seq_path_name = subj_id + "_" + seq_id
                rep_path_name = seq_path_name + "_" + rep_id

                data_tree_path = "/{}/{}/{}/" \
                    .format(subj_id, seq_path_name, rep_path_name)

                if not os.path.isdir(dataset_raw_path + data_tree_path):
                    # ignores data combinations which dont exist or are not available
                    #print("Requested data does not exist: {} | {} | {}"
                    #      .format(subj_id, seq_id, rep_id))
                    continue

                try:
                    '''
                    # (pointcloud) load extrinsic referential transforms + intrinsic camera params
                    extrinsic_calib_path = dataset_raw_path + "/{}/{}_extrinsic_calibration.json"\
                        .format(subj_id, subj_id)
                    
                    with open(extrinsic_calib_path, "r") as f:
                        extrinsic_ref_transforms = json.load(f)
                        
                    intrinsic_calib_path = dataset_raw_path + "/{}/{}_intrinsic_calibration.json"\
                        .format(subj_id, subj_id)
                    
                    with open(intrinsic_calib_path, "r") as f:
                        cam_intrinsics = json.load(f)
                        for k in cam_intrinsics:
                            cam_intrinsics[k] = np.array(cam_intrinsics[k])
                    '''

                    # load data alignment file
                    align_idx = pd.read_csv(dataset_proc_path + data_tree_path
                                            + "synchronized_data_idx.csv",
                                            index_col=0)

                    # load 3D skeleton data and reshape to [N, J, 3]
                    qpos3d_norm = pd.read_csv(dataset_proc_path + data_tree_path
                                              + "normalized_skeleton_3d.csv",
                                              index_col=0)
                    qpos3d_norm = qpos3d_norm.values.reshape(-1, len(xsens_joint_names), 3)

                    qpos3d_camref = pd.read_csv(dataset_proc_path + data_tree_path
                                                + "aligned_skeleton_3d.csv",
                                                index_col=0)
                    qpos3d_camref = qpos3d_camref.values.reshape(-1, len(xsens_joint_names), 3)

                    # load 2D skeleton data and reshape to [N, J, 2]
                    qpos2d_gait = pd.read_csv(dataset_proc_path + data_tree_path
                                              + "aligned_skeleton_2d_gait.csv",
                                              index_col=0)
                    qpos2d_gait = qpos2d_gait.values.reshape(-1, len(xsens_joint_names), 2)

                    qpos2d_posture = pd.read_csv(dataset_proc_path + data_tree_path
                                                 + "aligned_skeleton_2d_posture.csv",
                                                 index_col=0)
                    qpos2d_posture = qpos2d_posture.values.reshape(-1, len(xsens_joint_names), 2)

                    depth_g_path = dataset_raw_path + data_tree_path + "gait_depth_registered"
                    depth_p_path = dataset_raw_path + data_tree_path + "posture_depth_registered"

                    # get depth camera files
                    depth_fnum_pattern = re.compile("[a-zA-Z]+_[a-zA-Z]+_(\d+)_\d+_\d+.png")
                    depth_g_frame_ids = sorted(
                            os.listdir(depth_g_path),
                            key=lambda s: int(re.search(depth_fnum_pattern, s).group(1)))
                    depth_p_frame_ids = sorted(
                            os.listdir(depth_p_path),
                            key=lambda s: int(re.search(depth_fnum_pattern, s).group(1)))

                    for f_i, (f_dpt_p_idx, f_dpt_g_idx, _) \
                            in enumerate(zip(*align_idx.values.transpose())):

                        if not (f_i % undersample_rate == 0):
                            continue

                        # get 3D joint data
                        f_qpos3d_norm = qpos3d_norm[f_i]        # in normalized referential
                        f_qpos3d_camref = qpos3d_camref[f_i]    # in posture camera referential

                        # get 2D joint data
                        f_qpos2d_gait = qpos2d_gait[f_i]
                        f_qpos2d_posture = qpos2d_posture[f_i]

                        # example of how to get/visualize video frames,
                        # pointcloud and visualized overlaid keypoint data
                        f_dpt_g = process_depth_frame(
                                cv2.imread(depth_g_path + "/" + depth_g_frame_ids[f_dpt_g_idx],
                                           cv2.IMREAD_ANYDEPTH),
                                save_shape=frames_shape).astype(np.float32)

                        f_dpt_p = process_depth_frame(
                                cv2.imread(depth_p_path + "/" + depth_p_frame_ids[f_dpt_p_idx],
                                           cv2.IMREAD_ANYDEPTH),
                                save_shape=frames_shape).astype(np.float32)

                        '''
                        # extract pointcloud from multiple cameras and merge it
                        # no pointcloud or 3D joint data visualization as it needed 
                        # extra requirements for the repo, if this is really needed 
                        # feel free to request it.
                        pointcloud, pc_color = multiple_cams_merge_pointcloud(
                                depth_data=(f_dpt_p, f_dpt_g),
                                intrinsic_matrix=[cam_intrinsics["posture_camera"],
                                                  cam_intrinsics["gait_camera"]],
                                extrinsic_matrix=[np.eye(4),
                                                  extrinsic_ref_transforms[
                                                      "CamGaitToPostureTransform"]],
                                n_points=4096, depth_range=(0.1, 2.0))
                        '''
                        ################ visualize data ################
                        # resize keypoints 2D to match resized frames
                        f_qpos2d_data_gait_show = np.round(
                                f_qpos2d_gait * np.divide(frames_shape,
                                                          frames_original_shape)
                        ).astype(np.int32)

                        f_qpos2d_data_posture_show = np.round(
                                f_qpos2d_posture * np.divide(frames_shape,
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

                        ################################################

                except Exception as e:
                    if not ignore_error:
                        raise e
                    else:
                        print("Could not read data from: {} | {} | {}"
                              .format(subj_id, seq_id, rep_id),
                              " --------------------------------------->  Error: ", e)
    cv2.destroyAllWindows()


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

    read_walker_data(
            dataset_root_path=dataset_path,

            subj_ids=subj_ids,
            seq_ids=seq_ids,
            rep_ids=rep_ids,

            frames_shape=(320, 240),
            undersample_rate=1,
            ignore_error=True)
