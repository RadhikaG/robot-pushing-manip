import os
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from pydrake.all import (
    MeshcatVisualizerParams, ConstantVectorSource, Parser, Role, LeafSystem,
    JointStiffnessController, RotationMatrix, PlanarJoint, FixedOffsetFrame,
    ConstantVectorSource, PiecewisePolynomial, TrajectorySource, Multiplexer,
    PiecewisePose, AbstractValue, Sphere, Rgba
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
from RRTPlanner import CubeProblem, TreeNode, RRT, RRT_tools, IiwaProblem
from IKSolver import IKSolver
from IIWA_manipulation_station import IIWA_manipulation_station

class PushPlanner:
    # q_start and q_goal are for cube
    def __init__(self, q_start, q_goal, is_visualizing_rollouts=True, meshcat_instance=None):
        self.simulator = IIWA_manipulation_station(is_visualizing=is_visualizing_rollouts, meshcat_instance=meshcat_instance)
        self.meshcat = meshcat_instance

        self.cube_problem = CubeProblem(q_start, q_goal, self.simulator)
        self.rrt_tools_cube = RRT_tools(self.cube_problem)

        # just want to use utils for iiwa
        q_zero = np.zeros(7)
        self.iiwa_problem = IiwaProblem(q_zero, q_zero, 0, self.simulator)
        self.rrt_tools_iiwa = RRT_tools(self.iiwa_problem)
        self.ik_solver = IKSolver()

        np.random.seed(0)

        self.q_start = q_start
        self.q_goal = q_goal

        self.max_iterations = 100
        self.prob_sample_q_goal = 0.05
        self.cube_goal_epsilon = 0.1

        self.local_push_planner_max_iter = 7
        self.local_push_planner_tol = 0.1
        self.local_push_planner_num_pushes_per_sample = 4
        self.local_push_planner_extend_surface_pt_dist = 0.5

        self.is_visualizing_rollouts = is_visualizing_rollouts

    def l2_distance(self, q):
        x_diff_sq = q[0]**2
        y_diff_sq = q[1]**2
        angle_diff = min(abs(q[2]), 2*np.pi - abs(q[2]))
        angle_diff_sq = angle_diff**2
        return np.sqrt(x_diff_sq + y_diff_sq + angle_diff_sq)

    def rrt_cube_plan(self):
        q_start = self.q_start
        q_goal = self.q_goal
        rrt_tools_cube = self.rrt_tools_cube

        for k in range(self.max_iterations):
            q_sample = rrt_tools_cube.sample_node_in_configuration_space()
            rand_num = np.random.rand()
            if rand_num < self.prob_sample_q_goal:
                q_sample = q_goal
            n_near = rrt_tools_cube.find_nearest_node_in_RRT_graph(q_sample)
            intermediate_qs = rrt_tools_cube.calc_intermediate_qs_wo_collision(n_near.value, q_sample)

            last_node = n_near
            for n in range(1, len(intermediate_qs)):
                last_node = rrt_tools_cube.grow_rrt_tree(last_node, intermediate_qs[n])

            if np.allclose(last_node.value, q_goal):
                path = rrt_tools_cube.backup_path_from_node(last_node)
                return path

        return None

    def push_rrt_plan(self):
        rrt_tools_cube = self.rrt_tools_cube

        for k in range(self.max_iterations):
            print("push_rrt_plan: " + str(k))
            x_rand = rrt_tools_cube.sample_node_in_configuration_space()
            rand_num = np.random.rand()
            if rand_num < self.prob_sample_q_goal:
                x_rand = self.q_goal
            n_near = rrt_tools_cube.find_nearest_node_in_RRT_graph(x_rand)
            x_near = n_near.value

            pi = self.local_push_planner(x_near, x_rand)
            x_new, pi = self.new_state(x_rand, pi)

            last_node = rrt_tools_cube.grow_rrt_tree(n_near, x_new, pi)

            if self.l2_distance(last_node.value - self.q_goal) < self.cube_goal_epsilon or \
                    k == self.max_iterations-1:
                path = rrt_tools_cube.backup_node_path_from_node(last_node)
                push_q_seq = []
                for node in path[1:]:
                    pi = node.pi
                    for i in range(len(pi)):
                        push_q_seq.extend(pi[i][1])
                return push_q_seq

    def plot_rrt(self):
        rrt_tree = self.rrt_tools_cube.rrt_tree
        fig, ax = plt.subplots()
        root = rrt_tree.root
        nodes_to_plot = root.children
        while nodes_to_plot:
            node = nodes_to_plot.pop()
            parent = node.parent
            node_val = node.value
            parent_val = parent.value
            ax.plot([parent_val[0], node_val[0]], [parent_val[1], node_val[1]], "ko-", markersize=4, linewidth=1)
            print([parent_val, node_val])
            for child in node.children:
                nodes_to_plot.append(child)

        ax.plot(self.q_start[0], self.q_start[1], "rx", markersize=12, markeredgewidth=5)
        ax.plot(self.q_goal[0], self.q_goal[1], "gx", markersize=12, markeredgewidth=5)

        fig.savefig("out/rrt.png")
        plt.close(fig)


    def new_state(self, x_rand, pi):
        #dist_best = np.inf
        #x_best = None
        #for push in pi:
        #    x_push_end = push[0]
        #    dist_push_end = self.l2_distance(x_rand - x_push_end)
        #    if dist_push_end < dist_best:
        #        x_best = x_push_end
        #        dist_best = dist_push_end
        #return x_best

        #return pi[-1][0]

        dist_best = np.inf
        x_best = None
        push_index_best = -1
        for i,push in enumerate(pi):
            x_push_end = push[0]
            dist_push_end = self.l2_distance(x_rand - x_push_end)
            if dist_push_end < dist_best:
                x_best = x_push_end
                dist_best = dist_push_end
                push_index_best = i

        return x_best, pi[:push_index_best+1]

    def visualize_path(self, path):
        self.cube_problem.visualize_path(path)

    def local_push_planner(self, x_near, x_rand):
        print("local_push_planner")
        x = x_near
        # list of push sequences
        pi = []
        dist_best = np.inf

        i = 0
        while i < self.local_push_planner_max_iter:
            print(i)
            i += 1

            random_pushes = self.sample_random_actions(x)
            # u is actually series of configs in robot joint space, is a trajectory
            x_best, u_best, dist_best = self.select_action(x, random_pushes, x_rand)
            if u_best == None:
                continue
            pi.append((x_best, u_best, dist_best))
            x = x_best

            if dist_best < self.local_push_planner_tol:
                break

        return pi

    def random_point_on_face(self, face_id):
        box_x = 0.075
        box_y = 0.05
        box_z = 0.05
        if face_id == 0:
            p_CubeR = [-box_x/2, np.random.uniform(-box_y/2, box_y/2), np.random.uniform(0, box_z)]
        elif face_id == 1:
            p_CubeR = [np.random.uniform(-box_x/2, box_x/2), -box_y/2, np.random.uniform(0, box_z)]
        if face_id == 2:
            p_CubeR = [box_x/2, np.random.uniform(-box_y/2, box_y/2), np.random.uniform(0, box_z)]
        elif face_id == 3:
            p_CubeR = [np.random.uniform(-box_x/2, box_x/2), box_y/2, np.random.uniform(0, box_z)]
        return np.array(p_CubeR)

    ### config version
    #def sample_random_actions(self, x):
    #    print("sample_random_actions")
    #    random_pushes = []
    #    box_x = 0.075
    #    box_y = 0.05
    #    box_z = 0.05
    #    self.simulator.SetCubeConfig(x)

    #    for i in range(self.local_push_planner_num_pushes_per_sample):
    #        sel_face1 = np.random.randint(0, 4)
    #        sel_face2 = np.random.randint(0, 4)
    #        while sel_face2 == sel_face1:
    #            sel_face2 = np.random.randint(0, 4)

    #        p_CubeR1 = self.random_point_on_face(sel_face1)
    #        p_CubeR2 = self.random_point_on_face(sel_face2)

    #        m_R1R2 = p_CubeR1 - p_CubeR2

    #        p_CubeStart = p_CubeR1 - self.local_push_planner_extend_surface_pt_dist * m_R1R2
    #        p_CubeEnd = p_CubeR2 + self.local_push_planner_extend_surface_pt_dist * m_R1R2

    #        p_WStart = self.simulator.get_X_WCube() @ p_CubeStart + np.array([0, 0, 0.2])
    #        p_WEnd = self.simulator.get_X_WCube() @ p_CubeEnd + np.array([0, 0, 0.2])
    #        R_WPush = RotationMatrix.MakeXRotation(-np.pi/2)

    #        X_WStart = RigidTransform(R_WPush, p_WStart)
    #        X_WEnd = RigidTransform(R_WPush, p_WEnd)

    #        # hack to get cube out of the way
    #        self.simulator.SetCubeConfig([10, 10, 0])

    #        q_push_start, q_push_start_st = self.ik_solver.solve(X_WStart)
    #        q_push_end, q_push_end_st = self.ik_solver.solve(X_WEnd)

    #        self.simulator.SetIiwaConfiguration(q_push_start)
    #        print(q_push_start_st)
    #        time.sleep(5)
    #        self.simulator.SetIiwaConfiguration(q_push_end)
    #        print(q_push_end_st)
    #        time.sleep(5)

    #        # might need to replace below line with full RRT
    #        push_joint_config_seq = self.rrt_tools_iiwa.calc_intermediate_qs_wo_collision(q_push_start, q_push_end)

    #        # reset hack
    #        self.simulator.SetCubeConfig(x)

    #        print(push_joint_config_seq)
    #        if push_joint_config_seq:
    #            self.simulator.VisualizePushConfigSeq(push_joint_config_seq)
    #            time.sleep(10)
    #            random_pushes.append(push_joint_config_seq)

    #    return random_pushes

    ### pose version
    def sample_random_actions(self, x):
        print("sample_random_actions")
        random_pushes = []
        box_x = 0.075
        box_y = 0.05
        box_z = 0.05
        self.simulator.SetCubeConfig(x)

        for i in range(self.local_push_planner_num_pushes_per_sample):
            sel_face1 = np.random.randint(0, 4)
            sel_face2 = np.random.randint(0, 4)
            while sel_face2 == sel_face1:
                sel_face2 = np.random.randint(0, 4)

            X_WGStart = self.simulator.designPushPose(self.simulator.get_X_WContact(sel_face1))
            X_WGEnd = self.simulator.designPushPose(self.simulator.get_X_WContact(sel_face2))
            X_WStart = X_WGStart
            X_WEnd = RigidTransform(X_WGStart.rotation(), X_WGEnd.translation())

            m = X_WStart.translation() - X_WEnd.translation()

            d = 0.1
            p_WOffsetStart = X_WStart.translation() - d * m
            p_WOffsetEnd = X_WEnd.translation() + d * m

            push_pose_seq = []

            t = 0
            keyframe = p_WOffsetStart - t*m
            while np.linalg.norm(keyframe - p_WOffsetEnd) >= 0.01:
                push_pose_seq.append(RigidTransform(X_WStart.rotation(), keyframe))
                t += 0.01
                keyframe = p_WOffsetStart - t*m

            #if self.is_visualizing_rollouts:
                #self.simulator.visualize_frame("start", X_WStart)
                #self.simulator.visualize_frame("goal", X_WEnd)

            if push_pose_seq:
                #if self.is_visualizing_rollouts:
                #    self.simulator.VisualizePushConfigSeq(push_pose_seq)
                #    time.sleep(1)
                #    self.simulator.meshcat.Delete("pushing/")
                random_pushes.append(push_pose_seq)

        return random_pushes


    def select_action(self, x, random_pushes, x_rand):
        print("select_action")
        print(x)
        cost_best = np.inf
        x_best = None
        push_q_seq_best = None
        for push_q_seq in random_pushes:
            x_dash, cost_dash = self.simulate_action(x, push_q_seq, x_rand)
            if cost_dash < cost_best:
                x_best, push_q_seq_best, cost_best = x_dash, push_q_seq, cost_dash
        return x_best, push_q_seq_best, cost_best 

    ### pose version
    def simulate_action(self, x, push_pose_seq, x_rand):
        print("simulate_action")

        total_time = 5 
        times = np.linspace(0, total_time, len(push_pose_seq))
        push_traj = PiecewisePose.MakeLinear(times, push_pose_seq)

        default_iiwa_config, _ = self.ik_solver.solve(push_pose_seq[0])

        # adding offset to Z position so that object doesn't penetrate surface
        z_offset = np.array([0, 0, 0.01])
        default_cube_config = x + z_offset

        if self.is_visualizing_rollouts:
            #self.simulator.visualize_frame("bleh", push_pose_seq[0])
            #time.sleep(3)
            self.simulator.SetIiwaConfiguration(default_iiwa_config)
            time.sleep(3)
            self.simulator.VisualizePushConfigSeq(push_pose_seq)
            time.sleep(3)
            self.simulator.meshcat.Delete("pushing/")

        sim_station = IIWA_manipulation_station(
                is_visualizing=self.is_visualizing_rollouts, 
                meshcat_instance=self.meshcat, 
                traj=push_traj, 
                default_iiwa_config=default_iiwa_config, 
                default_cube_config=default_cube_config
        )
        
        try:
            if self.is_visualizing_rollouts:
                self.meshcat.SetObject("start", Sphere(0.01), Rgba(1, 0, 0, 1))
                self.meshcat.SetTransform("start", RigidTransform([self.q_start[0], self.q_start[1], 0.39]))
                self.meshcat.SetObject("goal", Sphere(0.01), Rgba(0, 1, 0, 1))
                self.meshcat.SetTransform("goal", RigidTransform([self.q_goal[0], self.q_goal[1], 0.39]))

                time.sleep(3)
                sim_station.SimulateStation(total_time, realtime=True)
                time.sleep(3)
            else:
                sim_station.SimulateStation(total_time)
        except RuntimeError:
            print("runtime error, continuing")

        x_dash = sim_station.GetCubeConfig(
                sim_station.plant.GetMyContextFromRoot(sim_station.simulator.get_context())
        )
        print(x_dash)

        cost_dash = self.l2_distance(x_rand - x_dash)

        print("simulate_action end")

        return x_dash, cost_dash
    ##

    ### config version
    #def simulate_action(self, x, push_q_seq, x_rand):
    #    print("simulate_action")
    #    
    #    total_time = 2
    #    breaks = np.arange(0, total_time, 1)
    #    samples = np.column_stack(push_q_seq)
    #    push_traj = PiecewisePolynomial.FirstOrderHold(breaks, samples)

    #    sim_station = IIWA_manipulation_station(is_visualizing=True, meshcat_instance=self.meshcat, traj=push_traj)
    #    # adding offset to Z position so that object doesn't penetrate surface
    #    z_offset = np.array([0, 0, 0.05])
    #    sim_station.SetCubeConfig(x + z_offset)
    #    sim_station.SimulateStation(breaks[-1]+1)

    #    x_dash = sim_station.GetCubeConfig()

    #    cost_dash = self.l2_distance(x_rand - x_dash)

    #    print("simulate_action end")

    #    return x_dash, cost_dash
    ###




