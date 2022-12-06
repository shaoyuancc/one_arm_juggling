import numpy as np
from IPython.display import clear_output

from pydrake.all import (AddMultibodyPlantSceneGraph, BsplineTrajectory,
                         DiagramBuilder, KinematicTrajectoryOptimization,
                         MeshcatVisualizer, MeshcatVisualizerParams,
                         MinimumDistanceConstraint, Parser, PositionConstraint,
                         Rgba, RigidTransform, Role, Solve, Sphere,
                         StartMeshcat, JacobianWrtVariable, RollPitchYaw,
                         JointSliders, RotationMatrix,
                         InverseKinematics,
                         LeafSystem, AbstractValue,
                         ConnectContactResultsToDrakeVisualizer, ContactResults, ContactVisualizerParams,
                         ContactVisualizer, Simulator, FixedOffsetFrame
                        )

from manipulation import running_as_notebook
from manipulation.meshcat_utils import (PublishPositionTrajectory, WsgButton,
                                        MeshcatPoseSliders)
from manipulation.scenarios import *
from manipulation.utils import AddPackagePaths, FindResource

from collections import namedtuple
from functools import partial

from IPython.display import HTML, Javascript, display
from pydrake.common.value import AbstractValue
from pydrake.geometry import (Cylinder, MeshcatVisualizer,
                              MeshcatVisualizerParams, Rgba, Role, Sphere)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import BodyIndex, JointIndex
from pydrake.perception import BaseField, Fields, PointCloud
from pydrake.solvers.mathematicalprogram import BoundingBoxConstraint
from pydrake.systems.framework import (DiagramBuilder, EventStatus, LeafSystem,
                                       PublishEvent, VectorSystem)
                    
def calculate_ball_vels(p1, p2, height):
    # p1: (x, y, z), ndarray
    # p2: (x, y, z), ndarray
    # height, pos real number

    g = 9.80665

    t = np.sqrt(2 * height / g)

    vx = (p2[0] - p1[0]) / (2 * t)
    vy = (p2[1] - p1[1]) / (2 * t)
    vz = t * g

    throw_vel = np.array([vx, vy, vz])
    catch_vel = np.array([vx, vy, -vz])
    total_duration =  2 * t
    return (throw_vel, catch_vel, total_duration)

def catch_pv_from_ball_pvt(p_WBall, v_WBall, t_travel):

    g = 9.80665

    p_WCatch = p_WBall.copy()
    V_WCatch = v_WBall.copy()

    p_WCatch[0] += v_WBall[0] * t_travel
    p_WCatch[1] += v_WBall[1] * t_travel
    p_WCatch[2] += v_WBall[2] * t_travel - 1/2 * g * t_travel ** 2

    V_WCatch[2] -= g * t_travel

    return p_WCatch, V_WCatch

def SpatialVelToJointVelConverter():
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    iiwa = AddIiwa(plant)
    wsg = AddWsg(plant, iiwa, welded=True, sphere=True)
    gripper_frame = plant.GetFrameByName("body", wsg)
    plant.Finalize()

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    def convert_spatial_vel_to_joint_vel(q, V_Gdesired):
        plant.SetPositions(plant_context, q)
        diagram.Publish(context)

        J_G = plant.CalcJacobianTranslationalVelocity(plant_context,
                                                      JacobianWrtVariable.kQDot,
                                                      gripper_frame, [0, 0, 0],
                                                      plant.world_frame(),
                                                      plant.world_frame())
        # print("J_G = ")
        # print(
        #     np.array2string(J_G,
                            # formatter={'float': lambda x: "{:5.2f}".format(x)}))
        # print(np.shape(J_G))
        # print(np.shape())
        V_Gdesired = V_Gdesired.reshape(3,1)
        v = np.linalg.pinv(J_G).dot(V_Gdesired)
        # print("Joint velocities")
        # print(np.array2string(v,
        #                     formatter={'float': lambda x: "{:5.3f}".format(x)}))
        return v

    return convert_spatial_vel_to_joint_vel

def convert_spatial_vel_ball_to_joint_vel_arm(q, V_Bdesired, p_GB_G):
    # context = diagram.CreateDefaultContext()
    # plant = diagram.GetSubsystemByName("plant")
    # plant_context = plant.GetMyContextFromRoot(context)
    # for i in range(plant.num_model_instances()):
    #     model_instance = ModelInstanceIndex(i)
    #     model_instance_name = plant.GetModelInstanceName(model_instance)
    #     if model_instance_name == "wsg":
    #         wsg = model_instance
    # gripper_frame = plant.GetFrameByName("body", wsg)

    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    iiwa = AddIiwa(plant)
    wsg = AddWsg(plant, iiwa, welded=True, sphere=True)
    gripper_frame = plant.GetFrameByName("body", wsg)
    plant.Finalize()

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    plant.SetPositions(plant_context, q)
    diagram.ForcedPublish(context)

    J_G = plant.CalcJacobianTranslationalVelocity(plant_context,
                                                    JacobianWrtVariable.kQDot,
                                                    gripper_frame, p_GB_G,
                                                    plant.world_frame(),
                                                    plant.world_frame())
    # print("J_G = ")
    # print(
    #     np.array2string(J_G,
                        # formatter={'float': lambda x: "{:5.2f}".format(x)}))
    # print(np.shape(J_G))
    # print(np.shape())
    V_Bdesired = V_Bdesired.reshape(3,1)
    v = np.linalg.pinv(J_G).dot(V_Bdesired)
    # print("Joint velocities")
    # print(np.array2string(v,
    #                     formatter={'float': lambda x: "{:5.3f}".format(x)}))
    return v

# def convert_spatial_postion_ball_to_joint_position_arm(q0, p_Bdesired, p_GB_G):
#     builder = DiagramBuilder()

#     plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
#     iiwa = AddIiwa(plant)
#     wsg = AddWsg(plant, iiwa, welded=True, sphere=True)
#     gripper_frame = plant.GetFrameByName("body", wsg)
#     plant.Finalize()

#     diagram = builder.Build()
#     context = diagram.CreateDefaultContext()
#     plant_context = plant.GetMyContextFromRoot(context)

#     plant.SetPositions(plant_context, q0)
#     diagram.ForcedPublish(context)

#     J_G = plant.CalcJacobianPositionVector(plant_context,
#                                             gripper_frame, p_GB_G,
#                                             plant.world_frame(),
#                                             plant.world_frame())
#     # print("J_G = ")
#     # print(
#     #     np.array2string(J_G,
#                         # formatter={'float': lambda x: "{:5.2f}".format(x)}))
#     # print(np.shape(J_G))
#     # print(np.shape())
#     p_Bdesired = p_Bdesired.reshape(3,1)
#     q = np.linalg.pinv(J_G).dot(p_Bdesired)
#     print(q)
#     # print("Joint velocities")
#     # print(np.array2string(v,
#     #                     formatter={'float': lambda x: "{:5.3f}".format(x)}))
    # return q.reshape(7,)

def RecordInterval( start_time, end_time,
                    simulator,
                    root_context,
                    plant,
                    visualizer,
                    time_step=1.0 / 33.0):

    plant_context = plant.GetMyContextFromRoot(root_context)
    visualizer_context = visualizer.GetMyContextFromRoot(root_context)

    visualizer.StartRecording(False)

    for t in np.append(
            np.arange(start_time, end_time,
                      time_step), end_time):
        simulator.AdvanceTo(t)
        visualizer.ForcedPublish(visualizer_context)

    visualizer.StopRecording()
    visualizer.PublishRecording()

# Another diagram for the objects the robot "knows about": arm, gripper, cameras.  Think of this as the model in the robot's head.
def make_internal_model():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    AddPackagePaths(parser)
    parser.AddAllModelsFromFile("./models/one_arm_juggling.dmd.yaml")
    plant.Finalize()
    return builder.Build()

# THIS IS NOT CURRENTLY BEING USED, NOT NECESSARY
def AddBalls(builder,
            plant,
            scene_graph,
            model_instance_prefix="ball"):
    """
    Exports the position and velocity output ports for balls
    """

    for index in range(plant.num_model_instances()):
        model_instance_index = ModelInstanceIndex(index)
        model_name = plant.GetModelInstanceName(model_instance_index)

        if model_name.startswith(model_instance_prefix):
            # Export the position and velocity outputs
            builder.ExportOutput(plant.get_state_output_port(model_instance_index),
                                 f"{model_name}_state")

def MyMakeManipulationStation(model_directives=None,
                            filename=None,
                            time_step=0.002,
                            iiwa_prefix="iiwa",
                            wsg_prefix="wsg",
                            camera_prefix="camera",
                            ball_prefix="ball",
                            prefinalize_callback=None,
                            package_xmls=[]):
    """
    Creates a manipulation station system, which is a sub-diagram containing:
      - A MultibodyPlant with populated via the Parser from the
        `model_directives` argument AND the `filename` argument.
      - A SceneGraph
      - For each model instance starting with `iiwa_prefix`, we add an
        additional iiwa controller system
      - For each model instance starting with `wsg_prefix`, we add an
        additional schunk controller system
      - For each body starting with `camera_prefix`, we add a RgbdSensor

    Args:
        builder: a DiagramBuilder

        model_directives: a string containing any model directives to be parsed

        filename: a string containing the name of an sdf, urdf, mujoco xml, or
        model directives yaml file.

        time_step: the standard MultibodyPlant time step.

        iiwa_prefix: Any model instances starting with `iiwa_prefix` will get
        an inverse dynamics controller, etc attached

        wsg_prefix: Any model instance starting with `wsg_prefix` will get a
        schunk controller

        camera_prefix: Any bodies in the plant (created during the
        plant_setup_callback) starting with this prefix will get a camera
        attached.

        prefinalize_callback: A function, setup(plant), that will be called
        with the multibody plant before calling finalize.  This can be useful
        for e.g. adding additional bodies/models to the simulation.

        package_xmls: A list of filenames to be passed to
        PackageMap.AddPackageXml().  This is useful if you need to add more
        models to your path (e.g. from your current working directory).
    """
    builder = DiagramBuilder()

    # Add (only) the iiwa, WSG, and cameras to the scene.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder,
                                                     time_step=time_step)
    parser = Parser(plant)
    for p in package_xmls:
        parser.package_map().AddPackageXml(p)
    AddPackagePaths(parser)
    if model_directives:
        directives = LoadModelDirectivesFromString(model_directives)
        ProcessModelDirectives(directives, parser)
    if filename:
        parser.AddAllModelsFromFile(filename)
    if prefinalize_callback:
        prefinalize_callback(plant)
    plant.Finalize()

    for i in range(plant.num_model_instances()):
        model_instance = ModelInstanceIndex(i)
        model_instance_name = plant.GetModelInstanceName(model_instance)

        if model_instance_name.startswith(iiwa_prefix):
            num_iiwa_positions = plant.num_positions(model_instance)

            # I need a PassThrough system so that I can export the input port.
            iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
            builder.ExportInput(iiwa_position.get_input_port(),
                                model_instance_name + "_position")
            builder.ExportOutput(iiwa_position.get_output_port(),
                                 model_instance_name + "_position_commanded")

            # Export the iiwa "state" outputs.
            demux = builder.AddSystem(
                Demultiplexer(2 * num_iiwa_positions, num_iiwa_positions))
            builder.Connect(plant.get_state_output_port(model_instance),
                            demux.get_input_port())
            builder.ExportOutput(demux.get_output_port(0),
                                 model_instance_name + "_position_measured")
            builder.ExportOutput(demux.get_output_port(1),
                                 model_instance_name + "_velocity_estimated")
            builder.ExportOutput(plant.get_state_output_port(model_instance),
                                 model_instance_name + "_state_estimated")

            # Make the plant for the iiwa controller to use.
            controller_plant = MultibodyPlant(time_step=time_step)
            # TODO: Add the correct IIWA model (introspected from MBP)
            if plant.num_positions(model_instance) == 3:
                controller_iiwa = AddPlanarIiwa(controller_plant)
            else:
                controller_iiwa = AddIiwa(controller_plant)
            AddWsg(controller_plant, controller_iiwa, welded=True)
            X_7E = RigidTransform(RollPitchYaw(0, 0, 0), [0, 0, 0.2]) #[-0.019, 0, 0.22]
            controller_plant.AddFrame(
                FixedOffsetFrame("extra_frame", controller_plant.GetFrameByName("iiwa_link_7"), X_7E)
            )
            print("Added extra frame!")
            controller_plant.Finalize()

            # Add the iiwa controller
            iiwa_controller = builder.AddSystem(
                InverseDynamicsController(controller_plant,
                                          kp=[100] * num_iiwa_positions,
                                          ki=[1] * num_iiwa_positions,
                                          kd=[20] * num_iiwa_positions,
                                          has_reference_acceleration=False))
            iiwa_controller.set_name(model_instance_name + "_controller")
            builder.Connect(plant.get_state_output_port(model_instance),
                            iiwa_controller.get_input_port_estimated_state())

            # Add in the feed-forward torque
            adder = builder.AddSystem(Adder(2, num_iiwa_positions))
            builder.Connect(iiwa_controller.get_output_port_control(),
                            adder.get_input_port(0))
            # Use a PassThrough to make the port optional (it will provide zero
            # values if not connected).
            torque_passthrough = builder.AddSystem(
                PassThrough([0] * num_iiwa_positions))
            builder.Connect(torque_passthrough.get_output_port(),
                            adder.get_input_port(1))
            builder.ExportInput(torque_passthrough.get_input_port(),
                                model_instance_name + "_feedforward_torque")
            builder.Connect(adder.get_output_port(),
                            plant.get_actuation_input_port(model_instance))

            # Add discrete derivative to command velocities.
            desired_state_from_position = builder.AddSystem(
                StateInterpolatorWithDiscreteDerivative(
                    num_iiwa_positions,
                    time_step,
                    suppress_initial_transient=True))
            desired_state_from_position.set_name(
                model_instance_name + "_desired_state_from_position")
            builder.Connect(desired_state_from_position.get_output_port(),
                            iiwa_controller.get_input_port_desired_state())
            builder.Connect(iiwa_position.get_output_port(),
                            desired_state_from_position.get_input_port())

            # Export commanded torques.
            builder.ExportOutput(adder.get_output_port(),
                                 model_instance_name + "_torque_commanded")
            builder.ExportOutput(adder.get_output_port(),
                                 model_instance_name + "_torque_measured")

            builder.ExportOutput(
                plant.get_generalized_contact_forces_output_port(
                    model_instance), model_instance_name + "_torque_external")

        elif model_instance_name.startswith(wsg_prefix):

            # Wsg controller.
            wsg_controller = builder.AddSystem(SchunkWsgPositionController())
            wsg_controller.set_name(model_instance_name + "_controller")
            builder.Connect(wsg_controller.get_generalized_force_output_port(),
                            plant.get_actuation_input_port(model_instance))
            builder.Connect(plant.get_state_output_port(model_instance),
                            wsg_controller.get_state_input_port())
            builder.ExportInput(
                wsg_controller.get_desired_position_input_port(),
                model_instance_name + "_position")
            builder.ExportInput(wsg_controller.get_force_limit_input_port(),
                                model_instance_name + "_force_limit")
            wsg_mbp_state_to_wsg_state = builder.AddSystem(
                MakeMultibodyStateToWsgStateSystem())
            builder.Connect(plant.get_state_output_port(model_instance),
                            wsg_mbp_state_to_wsg_state.get_input_port())
            builder.ExportOutput(wsg_mbp_state_to_wsg_state.get_output_port(),
                                 model_instance_name + "_state_measured")
            builder.ExportOutput(wsg_controller.get_grip_force_output_port(),
                                 model_instance_name + "_force_measured")

    # Cameras.
    AddRgbdSensors(builder,
                   plant,
                   scene_graph,
                   model_instance_prefix=camera_prefix)
    
    # Balls
    # AddBalls(   builder,
    #             plant,
    #             scene_graph,
    #             model_instance_prefix=ball_prefix)

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(plant.get_contact_results_output_port(),
                         "contact_results")
    builder.ExportOutput(plant.get_state_output_port(),
                         "plant_continuous_state")
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")
    builder.ExportOutput(plant.get_body_spatial_velocities_output_port(), "body_spatial_velocities")

    diagram = builder.Build()
    diagram.set_name("ManipulationStation")
    return diagram


def MakeDirectManipulationStation(model_directives=None,
                            filename=None,
                            time_step=0.002,
                            iiwa_prefix="iiwa",
                            wsg_prefix="wsg",
                            camera_prefix="camera",
                            ball_prefix="ball",
                            prefinalize_callback=None,
                            package_xmls=[],
                            meshcat=None):
    """
    Creates a manipulation station system, which is a sub-diagram containing:
      - A MultibodyPlant with populated via the Parser from the
        `model_directives` argument AND the `filename` argument.
      - A SceneGraph
      - For each model instance starting with `iiwa_prefix`, we add an
        additional iiwa controller system
      - For each model instance starting with `wsg_prefix`, we add an
        additional schunk controller system
      - For each body starting with `camera_prefix`, we add a RgbdSensor

    Args:
        builder: a DiagramBuilder

        model_directives: a string containing any model directives to be parsed

        filename: a string containing the name of an sdf, urdf, mujoco xml, or
        model directives yaml file.

        time_step: the standard MultibodyPlant time step.

        iiwa_prefix: Any model instances starting with `iiwa_prefix` will get
        an inverse dynamics controller, etc attached

        wsg_prefix: Any model instance starting with `wsg_prefix` will get a
        schunk controller

        camera_prefix: Any bodies in the plant (created during the
        plant_setup_callback) starting with this prefix will get a camera
        attached.

        prefinalize_callback: A function, setup(plant), that will be called
        with the multibody plant before calling finalize.  This can be useful
        for e.g. adding additional bodies/models to the simulation.

        package_xmls: A list of filenames to be passed to
        PackageMap.AddPackageXml().  This is useful if you need to add more
        models to your path (e.g. from your current working directory).
    """
    builder = DiagramBuilder()

    # Add (only) the iiwa, WSG, and cameras to the scene.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder,
                                                     time_step=time_step)
    parser = Parser(plant)
    for p in package_xmls:
        parser.package_map().AddPackageXml(p)
    AddPackagePaths(parser)
    if model_directives:
        directives = LoadModelDirectivesFromString(model_directives)
        ProcessModelDirectives(directives, parser)
    if filename:
        parser.AddAllModelsFromFile(filename)
    if prefinalize_callback:
        prefinalize_callback(plant)
    plant.Finalize()

    # Visualize contacts
    cparams = ContactVisualizerParams()
    cparams.force_threshold = 1e-4
    cparams.newtons_per_meter = 2
    cparams.radius = 0.001
    cparams.publish_period = time_step
    contact_visualizer = ContactVisualizer.AddToBuilder(
        builder, plant, meshcat, cparams)

    for i in range(plant.num_model_instances()):
        model_instance = ModelInstanceIndex(i)
        model_instance_name = plant.GetModelInstanceName(model_instance)

        if model_instance_name.startswith(iiwa_prefix):
            num_iiwa_positions = plant.num_positions(model_instance)

            # I need a PassThrough system so that I can export the input port.
            iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
            builder.ExportInput(iiwa_position.get_input_port(),
                                model_instance_name + "_position")
            builder.ExportOutput(iiwa_position.get_output_port(),
                                 model_instance_name + "_position_commanded")

            # Export the iiwa "state" outputs.
            demux = builder.AddSystem(
                Demultiplexer(2 * num_iiwa_positions, num_iiwa_positions))
            builder.Connect(plant.get_state_output_port(model_instance),
                            demux.get_input_port())
            builder.ExportOutput(demux.get_output_port(0),
                                 model_instance_name + "_position_measured")
            builder.ExportOutput(demux.get_output_port(1),
                                 model_instance_name + "_velocity_estimated")
            builder.ExportOutput(plant.get_state_output_port(model_instance),
                                 model_instance_name + "_state_estimated")

            # Make the plant for the iiwa controller to use.
            controller_plant = MultibodyPlant(time_step=time_step)
            # TODO: Add the correct IIWA model (introspected from MBP)
            if plant.num_positions(model_instance) == 3:
                controller_iiwa = AddPlanarIiwa(controller_plant)
            else:
                controller_iiwa = AddIiwa(controller_plant)
            AddWsg(controller_plant, controller_iiwa, welded=True)
            controller_plant.Finalize()

            # Add the iiwa controller
            iiwa_controller = builder.AddSystem(
                InverseDynamicsController(controller_plant,
                                          kp=[100] * num_iiwa_positions,
                                          ki=[1] * num_iiwa_positions,
                                          kd=[20] * num_iiwa_positions,
                                          has_reference_acceleration=True))
            iiwa_controller.set_name(model_instance_name + "_controller")

            builder.Connect(plant.get_state_output_port(model_instance),
                            iiwa_controller.get_input_port_estimated_state())

            # Export desired state.
            builder.ExportInput(iiwa_controller.get_input_port_desired_state(),
                                model_instance_name + "_state_desired")
            
            # Export desired acceleration.
            builder.ExportInput(iiwa_controller.get_input_port_desired_acceleration(),
                                model_instance_name + "_acceleration_desired")

            # Add in the feed-forward torque
            adder = builder.AddSystem(Adder(2, num_iiwa_positions))
            builder.Connect(iiwa_controller.get_output_port_control(),
                            adder.get_input_port(0))
            # Use a PassThrough to make the port optional (it will provide zero
            # values if not connected).
            torque_passthrough = builder.AddSystem(
                PassThrough([0] * num_iiwa_positions))
            builder.Connect(torque_passthrough.get_output_port(),
                            adder.get_input_port(1))
            builder.ExportInput(torque_passthrough.get_input_port(),
                                model_instance_name + "_feedforward_torque")
            builder.Connect(adder.get_output_port(),
                            plant.get_actuation_input_port(model_instance))

            # Export commanded torques.
            builder.ExportOutput(adder.get_output_port(),
                                 model_instance_name + "_torque_commanded")
            builder.ExportOutput(adder.get_output_port(),
                                 model_instance_name + "_torque_measured")

            builder.ExportOutput(
                plant.get_generalized_contact_forces_output_port(
                    model_instance), model_instance_name + "_torque_external")

        elif model_instance_name.startswith(wsg_prefix):

            # Wsg controller.
            wsg_controller = builder.AddSystem(SchunkWsgPositionController())
            wsg_controller.set_name(model_instance_name + "_controller")
            builder.Connect(wsg_controller.get_generalized_force_output_port(),
                            plant.get_actuation_input_port(model_instance))
            builder.Connect(plant.get_state_output_port(model_instance),
                            wsg_controller.get_state_input_port())
            builder.ExportInput(
                wsg_controller.get_desired_position_input_port(),
                model_instance_name + "_position")
            builder.ExportInput(wsg_controller.get_force_limit_input_port(),
                                model_instance_name + "_force_limit")
            wsg_mbp_state_to_wsg_state = builder.AddSystem(
                MakeMultibodyStateToWsgStateSystem())
            builder.Connect(plant.get_state_output_port(model_instance),
                            wsg_mbp_state_to_wsg_state.get_input_port())
            builder.ExportOutput(wsg_mbp_state_to_wsg_state.get_output_port(),
                                 model_instance_name + "_state_measured")
            builder.ExportOutput(wsg_controller.get_grip_force_output_port(),
                                 model_instance_name + "_force_measured")

    # Cameras.
    AddRgbdSensors(builder,
                   plant,
                   scene_graph,
                   model_instance_prefix=camera_prefix)
    
    # Balls
    # AddBalls(   builder,
    #             plant,
    #             scene_graph,
    #             model_instance_prefix=ball_prefix)

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(plant.get_contact_results_output_port(),
                         "contact_results")
    builder.ExportOutput(plant.get_state_output_port(),
                         "plant_continuous_state")
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")
    builder.ExportOutput(plant.get_body_spatial_velocities_output_port(), "body_spatial_velocities")

    diagram = builder.Build()
    diagram.set_name("ManipulationStation")
    return diagram

class PrintContactResults(LeafSystem):
    """ Helpers for printing contact results
    """
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareAbstractInputPort("contact_results",
                                      AbstractValue.Make(ContactResults()))
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.001, 0.0, self.Publish)

    def Publish(self, context, state):
        formatter = {'float': lambda x: '{:5.2f}'.format(x)}
        results = self.get_input_port().Eval(context)
        print(f"PrintContactResults at time {context.get_time()}")
        if results.num_point_pair_contacts()==0:
            print("no contact")
        for i in range(results.num_point_pair_contacts()):
            info = results.point_pair_contact_info(i)
            pair = info.point_pair()
            force_string = np.array2string(
                info.contact_force(), formatter=formatter)
            print(
              f"Pair ({i}) "
              f"slip speed:{info.slip_speed():.4f}, "
              f"depth:{pair.depth:.4f}, "
              f"force:{force_string}\n")
        clear_output(wait=True)

class MyMeshcatPoseSliders(LeafSystem):
    """
    Provides a set of ipywidget sliders (to be used in a Jupyter notebook) with
    one slider for each of roll, pitch, yaw, x, y, and z.  This can be used,
    for instance, as an interface to teleoperate the end-effector of a robot.

    .. pydrake_system::

        name: PoseSliders
        input_ports:
        - pose (optional)
        output_ports:
        - pose

    The optional `pose` input port is used ONLY at initialization; it can be
    used to set the initial pose e.g. from the current pose of a MultibodyPlant
    frame.
    """
    # TODO(russt): Use namedtuple defaults parameter once we are Python >= 3.7.
    Visible = namedtuple("Visible", ("roll", "pitch", "yaw", "x", "y", "z"))
    Visible.__new__.__defaults__ = (True, True, True, True, True, True)
    MinRange = namedtuple("MinRange", ("roll", "pitch", "yaw", "x", "y", "z"))
    MinRange.__new__.__defaults__ = (-np.pi, -np.pi, -np.pi, -1.0, -1.0, -1.0)
    MaxRange = namedtuple("MaxRange", ("roll", "pitch", "yaw", "x", "y", "z"))
    MaxRange.__new__.__defaults__ = (np.pi, np.pi, np.pi, 1.0, 1.0, 1.0)
    Value = namedtuple("Value", ("roll", "pitch", "yaw", "x", "y", "z"))
    Value.__new__.__defaults__ = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    DecrementKey = namedtuple("DecrementKey",
                              ("roll", "pitch", "yaw", "x", "y", "z"))
    DecrementKey.__new__.__defaults__ = ("KeyQ", "KeyW", "KeyA", "KeyJ", "KeyI",
                                         "KeyO")
    IncrementKey = namedtuple("IncrementKey",
                              ("roll", "pitch", "yaw", "x", "y", "z"))
    IncrementKey.__new__.__defaults__ = ("KeyE", "KeyS", "KeyD", "KeyL", "KeyK",
                                         "KeyU")

    def __init__(self,
                 meshcat,
                 visible=Visible(),
                 min_range=MinRange(),
                 max_range=MaxRange(),
                 value=Value(),
                 decrement_keycode=DecrementKey(),
                 increment_keycode=IncrementKey(),
                 body_index=None):
        """
        Args:
            meshcat: A Meshcat instance.
            visible: An object with boolean elements for 'roll', 'pitch',
                     'yaw', 'x', 'y', 'z'; the intention is for this to be the
                     PoseSliders.Visible() namedtuple.  Defaults to all true.
            min_range, max_range, value: Objects with float values for 'roll',
                      'pitch', 'yaw', 'x', 'y', 'z'; the intention is for the
                      caller to use the PoseSliders.MinRange, MaxRange, and
                      Value namedtuples.  See those tuples for default values.
            body_index: if the body_poses input port is connected, then this
                        index determine which pose is used to set the initial
                        slider positions during the Initialization event.
        """
        LeafSystem.__init__(self)
        port = self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()),
            self.DoCalcOutput)

        self.DeclareAbstractInputPort("body_poses",
                                      AbstractValue.Make([RigidTransform()]))
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

        # The widgets themselves have undeclared state.  For now, we accept it,
        # and simply disable caching on the output port.
        # TODO(russt): consider implementing the more elaborate methods seen
        # in, e.g., LcmMessageSubscriber.
        port.disable_caching_by_default()

        self._meshcat = meshcat
        self._visible = visible
        self._value = list(value)
        self._body_index = body_index

        print("Keyboard Controls:")
        for i in range(6):
            if visible[i]:
                meshcat.AddSlider(min=min_range[i],
                                  max=max_range[i],
                                  value=value[i],
                                  step=0.001,
                                  name=value._fields[i],
                                  decrement_keycode=decrement_keycode[i],
                                  increment_keycode=increment_keycode[i])

    def __del__(self):
        for s in ['roll', 'pitch', 'yaw', 'x', 'y', 'z']:
            if visible[s]:
                self._meshcat.DeleteSlider(s)

    def SetPose(self, pose):
        """
        Sets the current value of the sliders.

        Args:
            pose: Any viable argument for the RigidTransform
                  constructor.
        """
        tf = RigidTransform(pose)
        self.SetRpy(RollPitchYaw(tf.rotation()))
        self.SetXyz(tf.translation())

    def SetRpy(self, rpy):
        """
        Sets the current value of the sliders for roll, pitch, and yaw.

        Args:
            rpy: An instance of drake.math.RollPitchYaw
        """
        self._value[0] = rpy.roll_angle()
        self._value[1] = rpy.pitch_angle()
        self._value[2] = rpy.yaw_angle()
        for i in range(3):
            if self._visible[i]:
                self._meshcat.SetSliderValue(self._visible._fields[i],
                                             self._value[i])

    def SetXyz(self, xyz):
        """
        Sets the current value of the sliders for x, y, and z.

        Args:
            xyz: A 3 element iterable object with x, y, z.
        """
        self._value[3:] = xyz
        for i in range(3, 6):
            if self._visible[i]:
                self._meshcat.SetSliderValue(self._visible._fields[i],
                                             self._value[i])

    def _update_values(self):
        changed = False
        for i in range(6):
            if self._visible[i]:
                old_value = self._value[i]
                self._value[i] = self._meshcat.GetSliderValue(
                    self._visible._fields[i])
                changed = changed or self._value[i] != old_value
        return changed

    def _get_transform(self):
        return RigidTransform(
            RollPitchYaw(self._value[0], self._value[1], self._value[2]),
            self._value[3:])

    def DoCalcOutput(self, context, output):
        """Constructs the output values from the sliders."""
        self._update_values()
        output.set_value(self._get_transform())

    def Initialize(self, context, discrete_state):
        if self.get_input_port().HasValue(context):
            if self._body_index is None:
                raise RuntimeError(
                    "If the `body_poses` input port is connected, then you "
                    "must also pass a `body_index` to the constructor.")
            self.SetPose(self.get_input_port().Eval(context)[self._body_index])
            return EventStatus.Succeeded()
        return EventStatus.DidNothing()

    def Run(self, publishing_system, root_context, callback):
        # Calls callback(root_context, pose), then
        # publishing_system.ForcedPublish() each time the sliders change value.
        if not running_as_notebook:
            return

        publishing_context = publishing_system.GetMyContextFromRoot(
            root_context)

        self._meshcat.AddButton("Stop PoseSliders", "Escape")
        while self._meshcat.GetButtonClicks("Stop PoseSliders") < 1:
            if self._update_values():
                callback(root_context, self._get_transform())
                publishing_system.ForcedPublish(publishing_context)
            time.sleep(.1)

        self._meshcat.DeleteButton("Stop PoseSliders")

def get_q_with_diffik(q_original, p_target):
    meshcat = StartMeshcat()
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
        translation: [0, 0, 0.2]
        rotation: !Rpy { deg: [0, 0, 0]}
    """
    time_step = 0.001
    station = builder.AddSystem(
        MyMakeManipulationStation(model_directives=model_directives, time_step=time_step))
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
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    plant_context = plant.GetMyMutableContextFromRoot(context)
    q0 = plant.GetPositions(plant_context)
    non_iiwa_q0 = q0[7:]
    plant.SetPositions(plant_context, np.concatenate((q_original, non_iiwa_q0)))
    # print(f"initial joint positions {plant.GetPositions(plant_context)[0:7]}")
    # print(f"planned initial joint position {q_original}")
    simulator.AdvanceTo(0.1)
    while simulator.get_context().get_time() < 3:
        teleop.SetXyz(p_target)
        simulator.AdvanceTo(simulator.get_context().get_time() + 1.0)
    
    meshcat.Delete()
    print(f"final spatial positions {teleop._get_transform().translation()}")
    print(f"planned final spatial position {p_target}")
    print(f"final joint positions {plant.GetPositions(plant_context)[:7]}")
    return plant.GetPositions(plant_context)[:7]