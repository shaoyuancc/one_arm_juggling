import numpy as np
from IPython.display import clear_output

from pydrake.all import (AddMultibodyPlantSceneGraph,
                         DiagramBuilder,
                         MeshcatVisualizer, MeshcatVisualizerParams,
                         Parser, PositionConstraint,
                         Rgba, RigidTransform, Role, Solve, Sphere,
                         StartMeshcat, RollPitchYaw,
                         JointSliders, RotationMatrix,
                         InverseKinematics,
                         LeafSystem, AbstractValue,
                         ContactResults, ContactVisualizerParams,
                         Simulator, FixedOffsetFrame
                        )

from manipulation.meshcat_utils import (PublishPositionTrajectory, WsgButton)
from manipulation.scenarios import *
from utils import *

class DiffIKAdjuster():
    def __init__(self, meshcat=None):
        if meshcat == None:
            self.meshcat = StartMeshcat()
        else:
            self.meshcat = meshcat
        
        time_step = 0.001
        builder = DiagramBuilder()
        station = builder.AddSystem(
            MyMakeManipulationStation(filename="./models/iiwa_and_wsg.dmd.yaml", time_step=time_step))
        plant = station.GetSubsystemByName("plant")
        controller_plant = station.GetSubsystemByName(
            "iiwa_controller").get_multibody_plant_for_control()

        # Add a meshcat visualizer.
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, station.GetOutputPort("query_object"), meshcat)
        meshcat.ResetRenderMode()
        meshcat.DeleteAddedControls()

        # Set up differential inverse kinematics.
        differential_ik = AddIiwaDifferentialIK(
            builder,
            controller_plant,
            frame=controller_plant.GetFrameByName("extra_frame"))

        builder.Connect(differential_ik.get_output_port(),
                        station.GetInputPort("iiwa_position"))
        builder.Connect(station.GetOutputPort("iiwa_state_estimated"),
                        differential_ik.GetInputPort("robot_state"))

        # Set up teleop widgets.
        teleop = builder.AddSystem(
            MyMeshcatPoseSliders(
                meshcat,
                min_range=MyMeshcatPoseSliders.MinRange(roll=0,
                                                    pitch=-0.5,
                                                    yaw=-np.pi,
                                                    x=-0.6,
                                                    y=-0.8,
                                                    z=0.0),
                max_range=MyMeshcatPoseSliders.MaxRange(roll=2 * np.pi,
                                                    pitch=np.pi,
                                                    yaw=np.pi,
                                                    x=0.8,
                                                    y=0.3,
                                                    z=1.1),
                body_index=plant.GetBodyByName("iiwa_link_7").index()))
        builder.Connect(teleop.get_output_port(0),
                        differential_ik.get_input_port(0))
        builder.Connect(station.GetOutputPort("body_poses"),
                        teleop.GetInputPort("body_poses"))

        wsg_teleop = builder.AddSystem(WsgButton(meshcat))
        builder.Connect(wsg_teleop.get_output_port(0),
                        station.GetInputPort("wsg_position"))

        diagram = builder.Build()
        self.plant = plant
        self.teleop = teleop
        self.simulator = Simulator(diagram)
        self.context = self.simulator.get_mutable_context()
        self.plant_context = plant.GetMyMutableContextFromRoot(self.context)
    
    def get_joint_pos_for_spatial_pos(self, q_original, p_target):
        q0 = self.plant.GetPositions(self.plant_context)
        non_iiwa_q0 = q0[7:]
        self.plant.SetPositions(self.plant_context, np.concatenate((q_original, non_iiwa_q0)))
        # print(f"initial joint positions {self.plant.GetPositions(self.plant_context)[0:7]}")
        # print(f"planned initial joint position {q_original}")
        current_time = self.simulator.get_context().get_time()
        while self.simulator.get_context().get_time() < current_time + 3:
            self.teleop.SetXyz(p_target)
            self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 1.0)
        
        # print(f"final spatial positions {self.teleop._get_transform().translation()}")
        # print(f"planned final spatial position {p_target}")
        # print(f"final joint positions {self.plant.GetPositions(self.plant_context)[:7]}")
        
        return self.plant.GetPositions(self.plant_context)[:7]