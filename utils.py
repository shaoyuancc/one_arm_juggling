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