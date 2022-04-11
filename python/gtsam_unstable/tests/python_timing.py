import gtsam
import numpy as np
from time import time
np.random.seed(0)

try:
    L = gtsam.symbol_shorthand.L
    X = gtsam.symbol_shorthand.X
    PT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.ones(2))
    PT_NOISE = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy.Create(1), PT_NOISE)
    LANDMARK_PRIOR = gtsam.noiseModel.Diagonal.Sigmas(0.1 * np.ones(3))
    POSE_PRIOR = gtsam.noiseModel.Diagonal.Sigmas(0.1 * np.ones(6))
except AttributeError:
    L = gtsam.symbol_shorthand_L
    X = gtsam.symbol_shorthand_X
    PT_NOISE = gtsam.noiseModel_Diagonal.Sigmas(np.ones(2))
    PT_NOISE = gtsam.noiseModel_Robust.Create(gtsam.noiseModel_mEstimator_Cauchy.Create(1), PT_NOISE)
    LANDMARK_PRIOR = gtsam.noiseModel_Diagonal.Sigmas(0.1 * np.ones(3))
    POSE_PRIOR = gtsam.noiseModel_Diagonal.Sigmas(0.1 * np.ones(6))
FOCAL = 1500
IMG_SIZE = (1280, 1080)
RADIUS = 20
LANDMARK_STD = 2 * np.ones(3)
IMG_STD = 3 * np.ones(2)
POSE_STD = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
NPOSES = 20
NLANDMARKS = 1000
K = gtsam.Cal3_S2(FOCAL, FOCAL, 0, 0, 0)


def generate_poses():
    poses = []
    thetas = np.arange(-np.pi / 2, 3 * np.pi / 2, 2 * np.pi / NPOSES)
    for theta in thetas:
        pose = np.eye(4)
        pose[:3, 3] = np.array([RADIUS * np.cos(theta), 0, RADIUS * np.sin(theta)])
        pose[[0, 0], [0, 2]] = (-np.sin(theta), -np.cos(theta))
        pose[[2, 2], [0, 2]] = (np.cos(theta), -np.sin(theta))
        poses.append(gtsam.Pose3(pose))
    return poses


if __name__ == '__main__':
    values = gtsam.Values()
    graph = gtsam.NonlinearFactorGraph()
    poses = generate_poses()
    landmarks = np.random.normal(np.zeros(3), LANDMARK_STD, (NLANDMARKS, 3))
    landmarks_noise = np.random.normal(np.zeros(3), LANDMARK_STD, (NLANDMARKS, 3))
    for idx, pose in enumerate(poses):
        diff = np.random.normal(np.zeros(6), POSE_STD)
        values.insert(X(idx), pose.retract(diff))
    for idx, landmark in enumerate(landmarks):
        added = False
        for jdx, pose in enumerate(poses):
            try:
                pose_landmark = pose.transformTo(landmark)
            except TypeError:
                pose_landmark = pose.transformTo(gtsam.Point3(landmark)).vector()
            pt = pose_landmark[:2] / pose_landmark[2] * FOCAL
            if (-IMG_SIZE[0] / 2 < pt[0] < IMG_SIZE[0] / 2) and ((-IMG_SIZE[1] / 2 < pt[1] < IMG_SIZE[1] / 2)):
                diff = np.random.normal(np.zeros(2), IMG_STD)
                measured = pt + diff
                try:
                    factor = gtsam.GenericProjectionFactorCal3_S2(measured, PT_NOISE, X(jdx), L(idx), K)
                except TypeError:
                    factor = gtsam.GenericProjectionFactorCal3_S2(gtsam.Point2(measured), PT_NOISE, X(jdx), L(idx), K)
                graph.push_back(factor)
                if not added:
                    values.insert(L(idx), gtsam.Point3(landmark + landmarks_noise[idx]))
                    added = True
    graph.push_back(gtsam.PriorFactorPose3(X(0), poses[0], POSE_PRIOR))
    graph.push_back(gtsam.PriorFactorPoint3(L(0), gtsam.Point3(landmarks[0]), LANDMARK_PRIOR))
    error0 = graph.error(values)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values)
    t1 = time()
    result = optimizer.optimize()
    print(f"optimizer ran: {optimizer.iterations()} iterations, time took: {time() - t1} seconds, initial error: {error0}, final error: {graph.error(result)}")
    pass
