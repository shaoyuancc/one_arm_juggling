import numpy as np
from pydrake.all import (AddMultibodyPlantSceneGraph, BsplineTrajectory,
                         DiagramBuilder, KinematicTrajectoryOptimization,
                         MeshcatVisualizer, MeshcatVisualizerParams,
                         MinimumDistanceConstraint, Parser, PositionConstraint,
                         Rgba, RigidTransform, Role, Solve, Sphere,
                         StartMeshcat, JacobianWrtVariable, RollPitchYaw,
                         JointSliders, RotationMatrix,
                         InverseKinematics,
                         LeafSystem, AbstractValue
                        )

from manipulation import running_as_notebook
from manipulation.meshcat_utils import (PublishPositionTrajectory,
                                        model_inspector,
                                        MeshcatPoseSliders)
from manipulation.scenarios import AddIiwa, AddPlanarIiwa, AddShape, AddWsg, AddMultibodyTriad 
from manipulation.utils import AddPackagePaths, FindResource


def calculate_ball_vels(p1, p2, height):
    # p1: (x, y, z), ndarray
    # p2: (x, y, z), ndarray
    # height, pos real number

    g = 9.8

    t = np.sqrt(2 * height / g)

    vx = (p2[0] - p1[0]) / (2 * t)
    vy = (p2[1] - p1[1]) / (2 * t)
    vz = t * g

    throw_vel = np.array([vx, vy, vz])
    catch_vel = np.array([vx, vy, -vz])
    total_duration =  2 * t
    return (throw_vel, catch_vel, total_duration)

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
    diagram.Publish(context)

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
        visualizer.Publish(visualizer_context)

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