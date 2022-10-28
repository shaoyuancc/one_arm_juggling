import matplotlib.pyplot as plt
import mpld3
import time
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

from painter import *

def main():
    # Start the visualizer.
    meshcat = StartMeshcat()

    # define center and radius
    radius = 0.1
    p0 = [0.45, 0.0, 0.4]
    R0 = RotationMatrix(np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]).T)
    X_WorldCenter = RigidTransform(R0, p0)

    num_key_frames = 1
    """
    you may use different thetas as long as your trajectory starts
    from the Start Frame above and your rotation is positive
    in the world frame about +z axis
    thetas = np.linspace(0, 2*np.pi, num_key_frames)
    """
    # thetas = np.linspace(0, 2*np.pi, num_key_frames)

    painter = IIWA_Painter(meshcat=meshcat)

    X_WG = painter.get_X_WG()
    X_WEnd = RigidTransform(X_WG.rotation(),[0,0.3,0] + X_WG.translation())
    painter.visualize_frame(meshcat, 'gripper_current', X_WG)
    painter.visualize_frame(meshcat, 'X_WEnd', X_WEnd)
    # check key frames instead of interpolated trajectory
    # def visualize_key_frames(frame_poses):
    #     for i, pose in enumerate(frame_poses):
    #         painter.visualize_frame(meshcat, 'frame_{}'.format(i), pose, length=0.05)
            
    # key_frame_poses = compose_circular_key_frames(thetas, X_WorldCenter, painter.get_X_WG(), radius)   
    # visualize_key_frames(key_frame_poses)
    key_frame_poses = compose_eliptical_key_frames(X_WG, X_WEnd)

    total_time = 10
    times = np.linspace(0, total_time, num_key_frames+1)
    traj = PiecewisePose.MakeLinear(times, key_frame_poses)

    painter = IIWA_Painter(meshcat=meshcat, traj=traj)
    time.sleep(5)
    print("starting to simulate!")

    painter.paint(sim_duration=total_time)

    print("Done simulating!")
    # while True:
    #     #Infinite loop
    #     pass

main()