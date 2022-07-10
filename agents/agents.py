from importlib import reload

import agents.collab_ddpg_agent
reload(agents.collab_ddpg_agent)
from agents.collab_ddpg_agent import CollabDDPGAgent


def get_agent(name, agent_config, state_size, action_size, seed):
    """returns the initialized agent
    
    Params
    ======
        name (str): name of agent
        agent_config (dict): configurations for the specified agent
        state_size (int or tuple): state space size
        action_size (int): action space size
        seed (int): random seed
    """
    
    if name == "ddpg":
        agent = CollabDDPGAgent(agent_config[name], state_size, action_size, seed)
    else:
        raise NotImplementedError(f"Agent ({name}) not found!")
        
    return agent