import matplotlib.pyplot as plt
import mpld3
import numpy as np
from IPython.display import HTML, display
from manipulation import running_as_notebook, FindResource
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import MakeManipulationStation
from pydrake.all import (AddMultibodyPlantSceneGraph, AngleAxis, BasicVector,
                         ConstantVectorSource, DiagramBuilder,
                         FindResourceOrThrow, Integrator, JacobianWrtVariable,
                         LeafSystem, MeshcatVisualizer,
                         MeshcatVisualizerParams, MultibodyPlant,
                         MultibodyPositionToGeometryPose, Parser,
                         PiecewisePose, Quaternion, RigidTransform,
                         RollPitchYaw, RotationMatrix, SceneGraph, Simulator,
                         StartMeshcat, TrajectorySource)

class IIWA_Painter():
    def __init__(self, meshcat, traj=None):
        builder = DiagramBuilder()
        model_directives = """
directives:
- add_model:
    name: iiwa
    file: package://drake/manipulation/models/iiwa_description/iiwa7/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [ 1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
- add_model:
    name: wsg
    file: package://drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90]}
- add_frame:
    name: bin0_origin
    X_PF:
      base_frame: world
      rotation: !Rpy { deg: [0.0, 0.0, 90.0 ]}
      translation: [-0.05, -0.5, -0.015]

- add_model:
    name: bin0
    file: package://drake/examples/manipulation_station/models/bin.sdf

- add_weld:
    parent: bin0_origin
    child: bin0::bin_base

- add_frame:
    name: bin1_origin
    X_PF:
      base_frame: world
      rotation: !Rpy { deg: [0.0, 0.0, 180.0 ]}
      translation: [0.5, 0.05, -0.015]

- add_model:
    name: bin1
    file: package://drake/examples/manipulation_station/models/bin.sdf

- add_weld:
    parent: bin1_origin
    child: bin1::bin_base
"""
        # set up the system of manipulation station
        self.station = MakeManipulationStation(filename="./models/one_arm_juggling.dmd.yaml")
        # filename="./models/one_arm_juggling.dmd.yaml"
        # model_directives = model_directives
        # filename="./models/cluttered.dmd.yaml"
        # filename=FindResource("models/clutter.dmd.yaml")

        builder.AddSystem(self.station)
        self.plant = self.station.GetSubsystemByName("plant")

        # optionally add trajectory source
        if traj is not None:
            traj_V_G = traj.MakeDerivative()
            V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
            self.controller = builder.AddSystem(
                PseudoInverseController(self.plant))
            builder.Connect(V_G_source.get_output_port(),
                            self.controller.GetInputPort("V_G"))

            self.integrator = builder.AddSystem(Integrator(7))
            builder.Connect(self.controller.get_output_port(),
                            self.integrator.get_input_port())
            builder.Connect(self.integrator.get_output_port(),
                            self.station.GetInputPort("iiwa_position"))
            builder.Connect(
                self.station.GetOutputPort("iiwa_position_measured"),
                self.controller.GetInputPort("iiwa_position"))

        params = MeshcatVisualizerParams()
        params.delete_on_initialization_event = False
        self.visualizer = MeshcatVisualizer.AddToBuilder(
            builder, self.station.GetOutputPort("query_object"), meshcat, params)

        wsg_position = builder.AddSystem(ConstantVectorSource([0.1]))
        builder.Connect(wsg_position.get_output_port(),
                        self.station.GetInputPort("wsg_position"))

        self.diagram = builder.Build()
        self.gripper_frame = self.plant.GetFrameByName('body')
        self.world_frame = self.plant.world_frame()

        context = self.CreateDefaultContext()
        self.diagram.Publish(context)

    def visualize_frame(self, name, X_WF, length=0.15, radius=0.006):
        """
        visualize imaginary frame that are not attached to existing bodies
        
        Input: 
            name: the name of the frame (str)
            X_WF: a RigidTransform to from frame F to world.
        
        Frames whose names already exist will be overwritten by the new frame
        """
        AddMeshcatTriad(meshcat, "painter/" + name,
                        length=length, radius=radius, X_PT=X_WF)

    def CreateDefaultContext(self):
        context = self.diagram.CreateDefaultContext()
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, context)
        station_context = self.diagram.GetMutableSubsystemContext(
            self.station, context)

        # provide initial states
        # q0 = np.array([ 1.50666193e-05,  1.56461165e-01, -3.12761069e-05,
        #                -1.22296976e+00, -6.99097287e-06,  1.91181157e+00, -2.16900985e-05])
        q0 = np.array([ 1.40666193e-05,  1.56461165e-01, -3.82761069e-05,
                       -1.32296976e+00, -6.29097287e-06,  1.61181157e+00, -2.66900985e-05])
        # set the joint positions of the kuka arm
        iiwa = self.plant.GetModelInstanceByName("iiwa")
        self.plant.SetPositions(plant_context, iiwa, q0)
        self.plant.SetVelocities(plant_context, iiwa, np.zeros(7))
        wsg = self.plant.GetModelInstanceByName("wsg")

        self.plant.SetPositions(plant_context, wsg, [-0.05, 0.05])
        self.plant.SetVelocities(plant_context, wsg, [0, 0])        

        if hasattr(self, 'integrator'):
            self.integrator.set_integral_value(
                self.integrator.GetMyMutableContextFromRoot(context), q0)

        return context


    def get_X_WG(self, context=None):

        if not context:
            context = self.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        X_WG = self.plant.CalcRelativeTransform(
                    plant_context,
                    frame_A=self.world_frame,
                    frame_B=self.gripper_frame)
        return X_WG

    def paint(self, sim_duration=20.0):
        context = self.CreateDefaultContext()
        simulator = Simulator(self.diagram, context)
        simulator.set_target_realtime_rate(1.0)

        duration = sim_duration if running_as_notebook else 0.01
        simulator.AdvanceTo(duration)

class PseudoInverseController(LeafSystem):
    """
    same controller seen in-class
    """
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_G", BasicVector(6))
        self.q_port = self.DeclareVectorInputPort("iiwa_position", BasicVector(7))
        self.DeclareVectorOutputPort("iiwa_velocity", BasicVector(7),
                                     self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context, output):
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context, JacobianWrtVariable.kV,
            self._G, [0,0,0], self._W, self._W)
        J_G = J_G[:,self.iiwa_start:self.iiwa_end+1] # Only iiwa terms.
        v = np.linalg.pinv(J_G).dot(V_G) #important
        output.SetFromVector(v)

def compose_circular_key_frames(thetas, X_WorldCenter, X_WorldGripper_init, radius):
    """    
    returns: a list of RigidTransforms
    """
    ## this is an template, replace your code below
    key_frame_poses_in_world = [X_WorldGripper_init]
    # key_frame_poses_in_world = []

    p_CG1_C = [0,0,-radius]
    # X_CG1 = RigidTransform(RotationMatrix(), p_CG1_C)
    # X_WG1 = X_WorldCenter @ X_CG1
    for theta in thetas:
        R_CGn = RotationMatrix.MakeYRotation(theta)
        p_CGn_C = R_CGn @ p_CG1_C
        X_CGn = RigidTransform(R_CGn, p_CGn_C)
        X_WGn = X_WorldCenter @ X_CGn
        key_frame_poses_in_world.append(X_WGn)
        
    return key_frame_poses_in_world