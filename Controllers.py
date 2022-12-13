import numpy as np

from pydrake.all import (
    LeafSystem, AbstractValue, RigidTransform, RotationMatrix,
    JacobianWrtVariable
)

class PoseTrajectorySource(LeafSystem):
    def __init__(self, pose_trajectory):
        LeafSystem.__init__(self)
        self._pose_trajectory = pose_trajectory
        self.DeclareAbstractOutputPort(
                "pose", lambda: AbstractValue.Make(RigidTransform()),
                self.CalcPose)

    def CalcPose(self, context, output):
        output.set_value(self._pose_trajectory.GetPose(context.get_time()))

class TorqueController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()
        self._joint_indices = [
                plant.GetJointByName("iiwa_joint_"+str(j)).position_start()
                    for j in range(1, 8)
        ]

        self.DeclareVectorInputPort("iiwa_position_measured", 7)
        self.DeclareVectorInputPort("iiwa_cartesian_force_desired", 6)

        # for pure torque control mode on sim
        self.DeclareVectorOutputPort("iiwa_position_command", 7,
                self.CalcPositionOutput)
        self.DeclareVectorOutputPort("iiwa_torque_command", 7,
                self.CalcTorqueOutput)

    def CalcPositionOutput(self, context, output):
        # for pure torque control
        q_now = self.get_input_port(0).Eval(context)
        output.SetFromVector(q_now)

    def CalcTorqueOutput(self, context, output):
        q_now = self.get_input_port(0).Eval(context)
        cart_force_desired = self.get_input_port(1).Eval(context)

        self._plant.SetPositions(self._plant_context, self._iiwa, q_now)

        # cartesian quantities for eef
        X_now = self._plant.CalcRelativeTransform(
                self._plant_context, 
                self._W,
                self._G
        )

        rpy_now  = X_now.rotation()
        p_xyz_now = X_now.translation()

        # Bp's spatial velocity Jacobian in frame A with respect to "speeds", 6xN
        J_G = self._plant.CalcJacobianSpatialVelocity(
                self._plant_context, 
                JacobianWrtVariable.kQDot, # generalized positions (this) or generalized vels?
                self._G, # gripper frame
                [0,0,0], # origin of gripper frame - Bp
                self._W, # Bp's velocity in frame_A - frame_A
                self._W  # frame in which output is expressed - frame_E
        )

        J_G = J_G[:, self._joint_indices]

        tau_cmd = J_G.T.dot(cart_force_desired)
        output.SetFromVector(tau_cmd)
