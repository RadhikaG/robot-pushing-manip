import numpy as np

from pydrake.all import (
        FindResourceOrThrow, MultibodyPlant, Parser, RigidTransform, RollPitchYaw,
        RotationMatrix, Solve, SolutionResult, InverseKinematics, DiagramBuilder,
        AddMultibodyPlantSceneGraph
)
#from pydrake.multibody import inverse_kinematics
from manipulation.scenarios import AddIiwa, AddWsg

class IKSolver:
    def __init__(self):
        builder = DiagramBuilder()

        self.plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        iiwa = AddIiwa(self.plant)
        wsg = AddWsg(self.plant, iiwa, welded=True)
        self.plant.Finalize()
            
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(context)

        self.q0 = self.plant.GetPositions(self.plant_context)
        self.gripper_frame = self.plant.GetFrameByName("body", wsg)

    def solve(self, X_WG):
        ik = InverseKinematics(self.plant, self.plant_context)
        ik.AddPositionConstraint(
                self.gripper_frame, [0, 0, 0], self.plant.world_frame(), 
                X_WG.translation(), X_WG.translation()
        )
        ik.AddOrientationConstraint(
            self.gripper_frame, RotationMatrix(), self.plant.world_frame(), 
            X_WG.rotation(), 0.0
        )

        prog = ik.get_mutable_prog()
        q = ik.q()
        prog.AddQuadraticErrorCost(np.identity(len(q)), self.q0, q)
        prog.SetInitialGuess(q, self.q0)
        result = Solve(ik.prog())

        if result.get_solution_result() != SolutionResult.kSolutionFound:
            return result.GetSolution(q), False
        return result.GetSolution(q), True

#class IKSolver:
#    def __init__(self):
#        plant_iiwa = MultibodyPlant(0.0)
#        iiwa_file = FindResourceOrThrow(
#                "drake/manipulation/models/iiwa_description/iiwa7/iiwa7_no_collision.sdf"
#        )
#        iiwa = Parser(plant_iiwa).AddModelFromFile(iiwa_file)
#        # Define frames
#        world_frame = plant_iiwa.world_frame()
#        L0 = plant_iiwa.GetFrameByName("iiwa_link_0")
#        l7_frame = plant_iiwa.GetFrameByName("iiwa_link_7")
#        plant_iiwa.WeldFrames(world_frame, L0)
#        plant_iiwa.Finalize()
#        plant_context = plant_iiwa.CreateDefaultContext()
#
#        # gripper in link 7 frame
#        X_L7G = RigidTransform(rpy=RollPitchYaw([np.pi/2, 0, np.pi/2]), p=[0,0,0.114])
#        world_frame = plant_iiwa.world_frame()
#
#        self.world_frame = world_frame
#        self.l7_frame = l7_frame
#        self.plant_iiwa = plant_iiwa
#        self.plant_context = plant_context
#        self.X_L7G = X_L7G
#
#    def solve(self, X_WT, q_guess = None, theta_bound=0.01, position_bound=0.01):
#        """
#        plant: a mini plant only consists of iiwa arm with no gripper attached
#        X_WT: transform of target frame in world frame
#        q_guess: a guess on the joint state sol
#        """
#        plant = self.plant_iiwa
#        l7_frame = self.l7_frame
#        X_L7G = self.X_L7G
#        world_frame = self.world_frame
#
#        R_WT = X_WT.rotation()
#        p_WT = X_WT.translation()
#
#        if q_guess is None:
#            q_guess = np.zeros(7)
#
#        ik_instance = inverse_kinematics.InverseKinematics(plant)
#        # align frame A to frame B
#        ik_instance.AddOrientationConstraint(frameAbar=l7_frame, 
#                R_AbarA=X_L7G.rotation(),
#                #   R_AbarA=RotationMatrix(), # for link 7
#                frameBbar=world_frame, 
#                R_BbarB=R_WT, 
#                theta_bound=position_bound
#        )
#        # align point Q in frame B to the bounding box in frame A
#        ik_instance.AddPositionConstraint(frameB=l7_frame, 
#                p_BQ=X_L7G.translation(),
#                # p_BQ=[0,0,0], # for link 7
#                frameA=world_frame, 
#                p_AQ_lower=p_WT-position_bound, 
#                p_AQ_upper=p_WT+position_bound
#        )
#        prog = ik_instance.prog()
#        prog.SetInitialGuess(ik_instance.q(), q_guess)
#        result = Solve(prog)
#        if result.get_solution_result() != SolutionResult.kSolutionFound:
#            return result.GetSolution(ik_instance.q()), False
#        return result.GetSolution(ik_instance.q()), True



