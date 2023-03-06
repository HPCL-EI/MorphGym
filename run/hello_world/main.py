import morphgym as mg


agent,env = mg.make('Dog-2D','Plane',config=None) # agent name, env name, config path (optional)

for _ in range(50): # morph searching
    morphology = agent.morphology_space.sample()  # this is where you would insert your morph search algorithm
    agent.morph(morphology)
    observation, info = env.reset()

    for _ in range(500): # policy training
        action = agent.action_space.sample()
        observation, reward, terminated, info = env.step(action) # this is where you would insert your policy
        env.reset(terminated)
        env.render()
env.close()
