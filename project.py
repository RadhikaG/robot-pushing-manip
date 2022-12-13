import os
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from pydrake.all import (
    MeshcatVisualizerParams, ConstantVectorSource, Parser, Role, LeafSystem,
    JointStiffnessController, RotationMatrix, PlanarJoint, FixedOffsetFrame,
    ConstantVectorSource, PiecewisePolynomial, TrajectorySource, Multiplexer,
    PiecewisePose, AbstractValue
)
from pydrake.common import FindResourceOrThrow
from pydrake.geometry import MeshcatVisualizer, StartMeshcat
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import BodyIndex
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import (
        CameraInfo,
        RgbdSensor,
)
from pydrake.examples.manipulation_station import (
        IiwaCollisionModel,
        ManipulationStation,
)
from manipulation.scenarios import (
        MakeManipulationStation, AddIiwaDifferentialIK
)
from manipulation.meshcat_utils import AddMeshcatTriad
from RRTPlanner import CubeProblem, TreeNode, RRT, RRT_tools, IiwaProblem
from IKSolver import IKSolver
from PushPlanner import PushPlanner
from IIWA_manipulation_station import IIWA_manipulation_station

meshcat = StartMeshcat()

#def constructTraj(X_WGStart, X_WGEnd):
#    total_time = 1
#    times = np.linspace(0, total_time, 2)
#    pose_traj = PiecewisePose.MakeLinear(times, [X_WGStart, X_WGEnd])
#    return pose_traj
#
#if __name__ == "__main__":
#    ##
#    p_WGStart = np.array([-0.2, -0.2, 0.5])
#    offset = np.array([-0.1, -0.1, 0])
#    p_WGEnd = p_WGStart + offset
#
#    R_WG = RotationMatrix.MakeXRotation(-np.pi/2)
#    X_WGStart = RigidTransform(R_WG, p_WGStart)
#    X_WGEnd = RigidTransform(R_WG, p_WGEnd)
#    ##
#
#    ik_solver = IKSolver()
#    q_start, success = ik_solver.solve(X_WGStart)
#    
#    env = IIWA_manipulation_station(is_visualizing=True, meshcat_instance=meshcat, default_iiwa_config=q_start)
#    env.SimulateStation(20, realtime=True)
#    time.sleep(10)
#
#    #print(success)
#
#    #AddMeshcatTriad(meshcat, "start", length=0.15, radius=0.006, X_PT=X_WGStart)
#    #AddMeshcatTriad(meshcat, "end", length=0.15, radius=0.006, X_PT=X_WGEnd)
#
#    #pose_traj = constructTraj(X_WGStart, X_WGEnd)
#
#    #env = IIWA_manipulation_station(is_visualizing=True, meshcat_instance=meshcat, traj=pose_traj)
#
#    #p_WCube = np.array([-0.1, -0.3, 0.5])
#    #AddMeshcatTriad(meshcat, "cube", length=0.15, radius=0.006, X_PT=RigidTransform(R_WG, p_WCube))
#    #q_cube, success = ik_solver.solve(RigidTransform(R_WG, p_WCube))
#    #print(success)
#
#    #env.SetIiwaConfiguration(q_cube)
#    #time.sleep(5)
#    #env.SimulateStation(1, realtime=True)
#    #time.sleep(60)


if __name__ == "__main__": 
    env = IIWA_manipulation_station(is_visualizing=False, meshcat_instance=meshcat)
    X_WCube = env.get_X_WCube()
    R_WCube = X_WCube.rotation()
    p_WCube = X_WCube.translation()

    q_start = np.array([p_WCube[0], p_WCube[1], R_WCube.ToRollPitchYaw().yaw_angle()])
    q_goal = q_start + np.array([-0.2, -0.2, 0])

    push_planner = PushPlanner(q_start, q_goal, is_visualizing_rollouts=False, meshcat_instance=meshcat)
    push_q_seq = push_planner.push_rrt_plan()

    print("Done planning")

    push_planner.plot_rrt()

    total_time = 15
    times = np.linspace(0, total_time, len(push_q_seq))
    push_traj = PiecewisePose.MakeLinear(times, push_q_seq)

    ik_solver = IKSolver()
    default_iiwa_config, _ = ik_solver.solve(push_q_seq[0])
    default_cube_config = q_start

    env = IIWA_manipulation_station(
            is_visualizing=True, 
            meshcat_instance=meshcat,
            traj=push_traj, 
            default_iiwa_config=default_iiwa_config, 
            default_cube_config=default_cube_config
    )
    env.SimulateStation(total_time, realtime=True)


    ###
    #q_iiwa_start = np.array([-1.57, 0.1, 0, -1.2, 0, 1.6, 0])
    #q_iiwa_end = q_iiwa_start + np.array([0.2]*7)
    #samples = np.column_stack((q_iiwa_start, q_iiwa_end))
    #breaks = [0, 1]
    #traj = PiecewisePolynomial.FirstOrderHold(breaks, samples)
    #env = IIWA_manipulation_station(is_visualizing=True, traj=traj)
    #env.simulator.set_target_realtime_rate(1.0)
    #env.SimulateStation(2)
    ###

    #time.sleep(60)
