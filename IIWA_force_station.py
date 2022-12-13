import os
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import functools

from pydrake.all import (
    MeshcatVisualizerParams, ConstantVectorSource, Parser, Role, LeafSystem,
    JointStiffnessController, RotationMatrix, PlanarJoint, FixedOffsetFrame,
    ConstantVectorSource, PiecewisePolynomial, TrajectorySource, Multiplexer,
    PiecewisePose, AbstractValue
)
from pydrake.common import FindResourceOrThrow
from pydrake.geometry import MeshcatVisualizer, StartMeshcat
from pydrake.geometry.render import (
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    RenderCameraCore,
    RenderLabel,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
)
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
from Controllers import PoseTrajectorySource, TorqueController

def prefinalize_callback(plant):
    return

class IIWA_force_station:
    def __init__(self, is_visualizing, cartesian_force_desired=None, meshcat_instance=None, default_iiwa_config=None, default_cube_config=None):
        model_directives = open("model_directives.txt", "r").read()

        builder = DiagramBuilder()
        self.station = MakeManipulationStation(
                model_directives, 
                time_step=0.001, 
                package_xmls=["./package.xml"], 
                prefinalize_callback=prefinalize_callback
        )
        builder.AddSystem(self.station)

        self.plant = self.station.GetSubsystemByName("plant")

        self.scene_graph = self.station.GetSubsystemByName("scene_graph")

        # scene graph query output port.
        self.query_output_port = self.scene_graph.GetOutputPort("query")

        # wsg controller
        wsg_position = builder.AddSystem(ConstantVectorSource([0.01]))
        builder.Connect(wsg_position.get_output_port(),
                self.station.GetInputPort("wsg_position"))

        self.is_visualizing = is_visualizing
        if is_visualizing:
            self.meshcat = meshcat_instance
            params = MeshcatVisualizerParams()
            params.delete_on_initialization_event = False
            params.role = Role.kIllustration # kProximity for collisions, kIllustration for visual
            self.visualizer = MeshcatVisualizer.AddToBuilder(
                    builder, self.station.GetOutputPort("query_object"), self.meshcat, params)

        if cartesian_force_desired is not None:
            # setting up controller
            force_controller = builder.AddSystem(TorqueController(self.plant))
            builder.Connect(force_controller.get_output_port(0),
                    self.station.GetInputPort("iiwa_position"))
            builder.Connect(force_controller.get_output_port(1),
                    self.station.GetInputPort("iiwa_feedforward_torque"))

            builder.Connect(self.station.GetOutputPort("iiwa_position_measured"),
                    force_controller.get_input_port(0))

            cart_traj = PiecewisePolynomial.FirstOrderHold(np.arange(0, 20, 1), np.array([cartesian_force_desired]*20).T)
            traj_source = builder.AddSystem(TrajectorySource(cart_traj))
            builder.Connect(traj_source.get_output_port(),
                    force_controller.get_input_port(1))


        self.diagram = builder.Build()

        # contexts
        self.context_diagram = self.diagram.CreateDefaultContext()
        self.context_station = self.diagram.GetSubsystemContext(
                self.station, self.context_diagram)
        self.context_scene_graph = self.station.GetSubsystemContext(
                self.scene_graph, self.context_station)
        self.context_plant = self.station.GetMutableSubsystemContext(
                self.plant, self.context_station)

        #if cartesian_force_desired is not None:
        #    force_controller.get_input_port(1).FixValue(
        #            force_controller.GetMyMutableContextFromRoot(self.context_diagram), cartesian_force_desired)

        self.world_frame = self.plant.world_frame()
        self.gripper_frame = self.plant.GetFrameByName("body")
        self.cube_frame = self.plant.GetFrameByName("base_link")

        iiwa = self.plant.GetModelInstanceByName("iiwa")
        if default_iiwa_config is None:
            default_iiwa_config = np.array([-1.57, 0.1, 0, -1.2, 0, 1.6, 0])
        self.plant.SetDefaultPositions(
                iiwa,
                default_iiwa_config
        )
        if default_cube_config is None:
            default_cube_pose = RigidTransform(RotationMatrix(), np.array([0, -0.3, 0.39]))
        else:
            default_cube_pose = self.CubeConfigToTransform(default_cube_config)
        self.plant.SetDefaultFreeBodyPose(
                self.plant.GetBodyByName("base_link"),
                default_cube_pose
        )

        # mark initial configuration
        self.iiwa_q0 = self.plant.GetPositions(self.context_plant, iiwa)
        self.cube_q0 = self.GetCubeConfig()

        self.context_diagram = self.diagram.CreateDefaultContext()
        self.context_station = self.diagram.GetSubsystemContext(
                self.station, self.context_diagram)
        self.context_scene_graph = self.station.GetSubsystemContext(
                self.scene_graph, self.context_station)
        self.context_plant = self.station.GetMutableSubsystemContext(
                self.plant, self.context_station)

        self.simulator = Simulator(self.diagram)


    def CubeConfigToTransform(self, cube_q):
        R_WCube = RotationMatrix.MakeZRotation(cube_q[2])
        p_WCube = [cube_q[0], cube_q[1], self.get_X_WCube().translation()[2]]
        X_WCube = RigidTransform(R_WCube, p_WCube)
        return X_WCube

    def GetCubeConfig(self):
        X_WCube = self.get_X_WCube()
        R_WCube = X_WCube.rotation()
        p_WCube = X_WCube.translation()
        return np.array([p_WCube[0], p_WCube[1], R_WCube.ToRollPitchYaw().yaw_angle()])

    def SetCubeConfig(self, q):
        # x, y, theta to actual repr
        X_WCube = self.CubeConfigToTransform(q)

        #planar_cube_joint = self.plant.GetJointByName("cube_joint")
        #planar_cube_joint.set_pose(self.context_plant, [q[0], q[1]], q[2])
        
        #cube = self.plant.GetModelInstanceByName("cube")
        self.plant.SetFreeBodyPose(
                self.context_plant, 
                self.plant.GetBodyByName("base_link"), 
                X_WCube
        )

        self.diagram.Publish(self.context_diagram)

    def SetIiwaConfiguration(self, q_iiwa):
        iiwa = self.plant.GetModelInstanceByName("iiwa")
        self.plant.SetPositions(self.context_plant, iiwa, q_iiwa)
        self.diagram.Publish(self.context_diagram)

    def SetStationConfiguration(self, q_iiwa, q_cube):
        self.SetIiwaConfiguration(q_iiwa)
        self.SetCubeConfig(q_cube)

        self.diagram.Publish(self.context_diagram)

    def DrawStation(self, q_iiwa, q_cube):
        if not self.is_visualizing:
            print("not visualizing")
        self.SetStationConfiguration(q_iiwa, q_cube)
        self.diagram.Publish(self.context_diagram)

    def SimulateStation(self, advance_to_time, realtime=False):
        if self.is_visualizing:
            self.visualizer.StartRecording(True)
        if realtime:
            self.simulator.set_target_realtime_rate(1.0)

        self.simulator.AdvanceTo(advance_to_time)

        if self.is_visualizing:
            self.visualizer.PublishRecording()

    def ExistsCollision(self):
        query_object = self.query_output_port.Eval(self.context_scene_graph)
        collision_pairs = query_object.ComputePointPairPenetration()

        return len(collision_pairs) > 0

    def get_X_WG(self):
        X_WG = self.plant.CalcRelativeTransform(
                self.context_plant,
                frame_A=self.world_frame,
                frame_B=self.gripper_frame
        )
        return X_WG

    def get_X_WCube(self):
        X_WCube = self.plant.CalcRelativeTransform(
                self.context_plant,
                frame_A=self.world_frame,
                frame_B=self.cube_frame
        )
        return X_WCube

    def designPushPose(self, X_WContact):
        p_GContact = [0, 0.09, 0.0]
        R_GContact = RotationMatrix.MakeXRotation(np.pi/2)
        X_GContact = RigidTransform(R_GContact, p_GContact)
        X_ContactG = X_GContact.inverse()
        X_WG = X_WContact @ X_ContactG
        #return X_CubeContact, X_WG
        return X_WG

    def get_X_WContact(self, face_id):
        # Convention for each face:
        # Z/|\ 
        #   | /Y - Y points inside the box
        #   |/
        #   -----> X
        #
        # 0-west, 1-south, 2-east, 3-north, 4-top 
        # face_id refers to face of cube
        p_CubeContact = [0,0,0]
        R_WContact = RigidTransform()
        # box is 0.075 x 0.05 x 0.05
        box_x = 0.075
        box_y = 0.05
        box_z = 0.05
        offset = 0.03
        if face_id == 0:
            p_CubeContact = [-box_x/2 - offset, 0, box_z/2]
            R_CubeContact = RotationMatrix.MakeZRotation(3*np.pi/2)
        elif face_id == 1:
            p_CubeContact = [0, -box_y/2 - offset, box_z/2]
            R_CubeContact = RotationMatrix.MakeZRotation(0)
        elif face_id == 2:
            p_CubeContact = [+box_x/2 + offset, 0, box_z/2]
            R_CubeContact = RotationMatrix.MakeZRotation(np.pi/2)
        elif face_id == 3:
            p_CubeContact = [0, +box_y/2 + offset, box_z/2]
            R_CubeContact = RotationMatrix.MakeZRotation(np.pi)
        elif face_id == 4:
            p_CubeContact = [0, 0, box_z + offset]
            R_CubeContact = RotationMatrix.MakeXRotation(-np.pi/2)

        X_CubeContact = RigidTransform(R_CubeContact, p_CubeContact)

        X_WContact = self.get_X_WCube() @ X_CubeContact

        return X_WContact

    def visualize_frame(self, name, X_WF):
        AddMeshcatTriad(self.meshcat, "pushing/"+name, length=0.15, radius=0.006, X_PT=X_WF)

    ### pose version
    def VisualizePushConfigSeq(self, push_q_seq):
        wsg = self.plant.GetModelInstanceByName("wsg")
        gripper = self.plant.GetBodyByName("body", wsg)
        for i,push_q in enumerate(push_q_seq):
            self.visualize_frame("pose_{i}".format(i=i), push_q)

    ### config version
    #def VisualizePushConfigSeq(self, push_q_seq):
    #    wsg = self.plant.GetModelInstanceByName("wsg")
    #    gripper = self.plant.GetBodyByName("body", wsg)
    #    for i,push_q in enumerate(push_q_seq):
    #        self.SetIiwaConfiguration(push_q)
    #        pose = self.plant.EvalBodyPoseInWorld(self.context_plant, gripper)
    #        self.visualize_frame("pose_{i}".format(i=i), pose)



