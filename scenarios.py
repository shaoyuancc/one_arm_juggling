import numpy as np

import pydrake.all
from pydrake.all import *

#IIWA_DEFAULT_Q = [0.0, 0.1, 0, -1.2, 0, 1.6, 0]
TABLE_HEIGHT = 0.3
#PADDLE_RADIUS = 0.225

def AddCameras(plant, name = 'cameras'):
    parser = pydrake.multibody.parsing.Parser(plant)
    cameras = parser.AddModelFromFile("models/cameras.yaml", name)
    return cameras

# Adds a ball to the scene
def AddBall(plant, name='ball'):
    parser = pydrake.multibody.parsing.Parser(plant)
    ball = parser.AddModelFromFile("models/ball.sdf", name)
    return ball

def AddTable(plant, height=TABLE_HEIGHT, name='table'):
    parser = pydrake.multibody.parsing.Parser(plant)
    table = parser.AddModelFromFile("models/table.sdf", name)

    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("table"), 
        RigidTransform(p=[2, 0, height-0.5])
    )

    return table

# Sets up the ball with the given position and velocity
def SetupBall(scenario, context, pos=[0,0,0], vel=[0,0,0]):
    plant = scenario.plant
    ball = scenario.ball
    plant_context = plant.GetMyMutableContextFromRoot(context)

    plant.SetPositions(plant_context, ball, np.hstack((1,0,0,0,pos)))
    plant.SetVelocities(plant_context, ball, np.hstack((0,0,0,vel)))

def ConfigureBall(builder, plant, ball, name='ball'):
    demux = builder.AddSystem(Demultiplexer([4, 3, 3, 3]))
    builder.Connect(plant.get_state_output_port(ball), demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(1), f"{name}_position_measured")
    builder.ExportOutput(demux.get_output_port(3), f"{name}_velocity_estimated")
    builder.ExportOutput(plant.get_state_output_port(ball), f"{name}_state_estimated")

class Scenario():
    def __init__(self, time_step=0.002):
        self._builder = DiagramBuilder()

        self.plant, self._scene_graph = AddMultibodyPlantSceneGraph(
            self._builder, time_step=time_step)

        # Add two iiwas to the scene
        #self.iiwa1 = AddIiwa(self.plant, pose=RigidTransform(p=[-0.5,0,0]), name='iiwa1')
        #self.iiwa2 = AddIiwa(self.plant, pose=RigidTransform(RollPitchYaw(0, 0, np.pi), [4.5,0,0]), name='iiwa2')
        
        # Give the iiwas paddles
        # self.paddle1 = AddPaddle(self.plant, self.iiwa1, name='paddle1')
        # self.paddle2 = AddPaddle(self.plant, self.iiwa2, name='paddle2')
        
        # Add the ball and table
        self.ball = AddBall(self.plant)
        self.table = AddTable(self.plant)
        self.cameras = AddCameras(self.plant)

        self.plant.Finalize()

        # Configure the ports for the iiwas
        # ConfigureIiwa(self._builder, self.plant, self.iiwa1, name='iiwa1', time_step=time_step)
        # ConfigureIiwa(self._builder, self.plant, self.iiwa2, name='iiwa2', time_step=time_step)

        # Configure the ports for the ball
        ConfigureBall(self._builder, self.plant, self.ball)

        # Export "cheat" ports.
        self._builder.ExportOutput(self._scene_graph.get_query_output_port(), "geometry_query")
        self._builder.ExportOutput(self.plant.get_contact_results_output_port(), 
                            "contact_results")
        self._builder.ExportOutput(self.plant.get_state_output_port(), 
                            "plant_continuous_state")

    # builds diagram
    def get_diagram(self):
        diagram = self._builder.Build()
        return diagram
    
    # returns iiwa and paddle associated with name
    # def get_iiwa(self, name):
    #     if name == 'iiwa1':
    #         return self.iiwa1, self.paddle1
    #     elif name == 'iiwa2':
    #         return self.iiwa2, self.paddle2
