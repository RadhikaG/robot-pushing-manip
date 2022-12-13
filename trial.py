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
from IIWA_force_station import IIWA_force_station

meshcat = StartMeshcat()

if __name__ == "__main__":
    ##
    env = IIWA_force_station(is_visualizing=False)

    X_WContact = env.get_X_WContact(3)
    X_WG = env.designPushPose(X_WContact)

    ik = IKSolver()
    q_start, _ = ik.solve(X_WG)

    cartesian_force_desired = np.array([0, 0, 0, 0, -0.35, 0])

    env = IIWA_force_station(
            is_visualizing=True, 
            cartesian_force_desired=cartesian_force_desired, 
            meshcat_instance=meshcat, 
            default_iiwa_config=q_start)
    #env = IIWA_manipulation_station(is_visualizing=True, meshcat_instance=meshcat)
    env.diagram.Publish(env.context_diagram)
    env.visualize_frame("blah", X_WG)
    time.sleep(10)
    env.SimulateStation(4, realtime=True)
