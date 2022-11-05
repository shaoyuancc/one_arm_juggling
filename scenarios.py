# Adds a ball to the scene
def AddBall(plant, name='ball'):
    parser = pydrake.multibody.parsing.Parser(plant)
    ball = parser.AddModelFromFile("models/ball.sdf", name)
    return ball

# Sets up the ball with the given position and velocity
def SetupBall(scenario, context, pos=[0,0,0], vel=[0,0,0]):
    plant = scenario.plant
    ball = scenario.ball
    plant_context = plant.GetMyMutableContextFromRoot(context)

    plant.SetPositions(plant_context, ball, np.hstack((1,0,0,0,pos)))
    plant.SetVelocities(plant_context, ball, np.hstack((0,0,0,vel)))