import numpy as np
import time
from manipulation.exercises.trajectories.rrt_planner.robot import (
            ConfigurationSpace, Range)
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import \
            Problem
from pydrake.all import (
        FindResourceOrThrow, MultibodyPlant, Parser, RigidTransform, RollPitchYaw,
        RotationMatrix, Solve, SolutionResult
)
from pydrake.multibody import inverse_kinematics

class CubeProblem(Problem):
    def __init__(self, q_start, q_goal, collision_checker, is_visualizing=False):
        self.q_start = q_start
        self.q_goal = q_goal
        self.is_visualizing = is_visualizing

        self.collision_checker = collision_checker

        x_range = Range(-5, 5)
        y_range = Range(-5, 5)
        # no z yet
        theta_range = Range(-np.pi, np.pi)
        range_list = [x_range, y_range, theta_range]

        def l2_distance(q):
            x_diff_sq = q[0]**2
            y_diff_sq = q[1]**2
            angle_diff = min(abs(q[2]), 2*np.pi - abs(q[2]))
            angle_diff_sq = angle_diff**2
            return np.sqrt(x_diff_sq + y_diff_sq + angle_diff_sq)

        max_steps = [0.02, 0.02, np.pi/180 * 2]

        cspace_cube = ConfigurationSpace(range_list, l2_distance, max_steps)

        Problem.__init__(
                self,
                x=10,
                y=10,
                robot=None,
                obstacles=None,
                start=tuple(q_start),
                goal=tuple(q_goal),
                cspace=cspace_cube
        )

    def collide(self, configuration):
        return False
        #q = np.array(configuration)
        #return self.collision_checker.ExistsCollision(q, self.gripper_setpoint,
        #                                              self.left_door_angle,
        #                                              self.right_door_angle)

    def visualize_path(self, path):
        if path is not None:
            for q in path:
                self.collision_checker.DrawCubeConfig(q)
                time.sleep(0.2)

class IiwaProblem(Problem):
    def __init__(self,
                 q_start: np.array,
                 q_goal: np.array,
                 gripper_setpoint: float,
                 collision_checker,
                 is_visualizing=False):
        self.gripper_setpoint = gripper_setpoint
        self.is_visualizing = is_visualizing

        self.collision_checker = collision_checker

        # Construct configuration space for IIWA.
        plant = self.collision_checker.plant
        nq = 7
        joint_limits = np.zeros((nq, 2))
        for i in range(nq):
            joint = plant.GetJointByName("iiwa_joint_%i" % (i + 1))
            joint_limits[i, 0] = joint.position_lower_limits()
            joint_limits[i, 1] = joint.position_upper_limits()

        range_list = []
        for joint_limit in joint_limits:
            range_list.append(Range(joint_limit[0], joint_limit[1]))

        def l2_distance(q: tuple):
            sum = 0
            for q_i in q:
                sum += q_i**2
            return np.sqrt(sum)

        max_steps = nq * [np.pi / 180 * 2]  # three degrees
        cspace_iiwa = ConfigurationSpace(range_list, l2_distance, max_steps)

        # Call base class constructor.
        Problem.__init__(
            self,
            x=10,  # not used.
            y=10,  # not used.
            robot=None,  # not used.
            obstacles=None,  # not used.
            start=tuple(q_start),
            goal=tuple(q_goal),
            cspace=cspace_iiwa)

    def collide(self, configuration):
        q = np.array(configuration)
        self.collision_checker.SetIiwaConfiguration(configuration)
        return self.collision_checker.ExistsCollision()

class TreeNode:
    def __init__(self, value, parent=None, pi=None):
        self.value = value  # tuple of floats representing a configuration
        self.parent = parent  # another TreeNode
        self.children = []  # list of TreeNodes
        self.pi = pi # list of pushing iiwa configs from its parent -> self

class RRT:
    """
    RRT Tree.
    """
    def __init__(self, root: TreeNode, cspace):
        self.root = root  # root TreeNode
        self.cspace = cspace  # robot.ConfigurationSpace
        self.size = 1  # int length of path
        #self.max_recursion = 1000  # int length of longest possible path
        self.max_recursion = 1000  # int length of longest possible path

    def add_configuration(self, parent_node, child_value, child_pi=None):
        child_node = TreeNode(child_value, parent_node, child_pi)
        parent_node.children.append(child_node)
        self.size += 1
        return child_node

    # Brute force nearest, handles general distance functions
    def nearest(self, configuration):
        """
        Finds the nearest node by distance to configuration in the
        configuration space.

        Args:
            configuration: tuple of floats representing a configuration of a
            robot

        Returns:
            closest: TreeNode. the closest node in the configuration space
            to configuration
            distance: float. distance from configuration to closest
        """
        assert self.cspace.valid_configuration(configuration)
        def recur(node, depth=0):
            closest, distance = node, self.cspace.distance(node.value, configuration)
            if depth < self.max_recursion:
                for child in node.children:
                    (child_closest, child_distance) = recur(child, depth+1)
                    if child_distance < distance:
                        closest = child_closest
                        child_distance = child_distance
            return closest, distance
        return recur(self.root)[0]


class RRT_tools:
    def __init__(self, problem):
        # rrt is a tree 
        self.rrt_tree = RRT(TreeNode(problem.start), problem.cspace)
        problem.rrts = [self.rrt_tree]
        self.problem = problem

    def find_nearest_node_in_RRT_graph(self, q_sample):
        nearest_node = self.rrt_tree.nearest(q_sample)
        return nearest_node

    def sample_node_in_configuration_space(self):
        q_sample = self.problem.cspace.sample()
        return q_sample

    def calc_intermediate_qs_wo_collision(self, q_start, q_end):
        '''create more samples by linear interpolation from q_start
        to q_end. Return all samples that are not in collision

        Example interpolated path: 
        q_start, qa, qb, (Obstacle), qc , q_end
        returns >>> q_start, qa, qb
        '''
        return self.problem.safe_path(q_start, q_end)

    def grow_rrt_tree(self, parent_node, q_sample, q_sample_pi=None):
        """
        add q_sample to the rrt tree as a child of the parent node
        returns the rrt tree node generated from q_sample
        """
        child_node = self.rrt_tree.add_configuration(parent_node, q_sample, q_sample_pi)
        return child_node

    def node_reaches_goal(self, node):
        return node.value == self.problem.goal

    def backup_path_from_node(self, node):
        path = [node.value]
        while node.parent is not None:
            node = node.parent
            path.append(node.value)
            path.reverse()
        return path

    def backup_node_path_from_node(self, node):
        path = [node]
        while node.parent is not None:
            node = node.parent
            path.append(node)
            path.reverse()
        return path
