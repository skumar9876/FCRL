"""
Multi-Agent RL baseline implementation as described in "Federated Control with Hierarchical Multi-Agent RL" 
by Kumar et al. 
@author: Saurabh Kumar
"""

from dqn import DqnAgent
import numpy as np

class MultiRLAgent(object):

    def __init__(self, 
        num_agents,
        state_size,
        num_actions,
        num_communication_turns):
        """Initializes a multi-agent RL agent. Used as a baseline in the FCRL experiments.
        Args:
            num_agents: Number of agents / controllers.
            state_size: The base size of each agent's state.
            num_actions: The number of actions. 
            num_communication_turns: Number of turns for which the agents communicate.
        """

        self._num_agents = num_agents
        self._num_actions = num_actions
        self._num_communication_turns = num_communication_turns
        self._base_state_size = state_size
        self._state_size = state_size + self._num_communication_turns * self._num_actions

        # Initialize the agents.
        self._agents = []
        for i in xrange(self._num_agents):
            self._agents.append(
                DqnAgent(state_dims=self._state_size, 
                    num_actions=self._num_actions,
                    learning_rate=0.1,
                    epsilon_end=0.01))

        self._communication_states = None

    def sample(self, environment_state, controller_ordering, eval=False):
        """Return a sampled set of output actions from all agents.
           Args:
            environment_state: The environment state.
            controller_ordering: The desired ordering of controllers' outputs.
            eval: Whether or not this an evaluation episode.
           Returns:
            output_actions - a set of output actions from all agents.
        """
        # Split the environment state into num_agents states of equal size.
        agents_states = np.split(environment_state, self._num_agents)

        for i in xrange(self._num_agents):
            # Construct one-hot vector indicating the controller's position in the controller ordering.
            controller_ordering = list(controller_ordering)
            controller_position = controller_ordering.index(i)
            controller_ordering_vector = np.zeros(self._num_agents)
            controller_ordering_vector[controller_position] = 1
            # Add the controller ordering vector to the controller's state representation.
            agents_states[i].append(np.copy(controller_ordering_vector))
            # Add the communication turn vector (placeholder containing zeros) to the controller's state representation.
            agents_states[i].append(
                np.zeros(self._num_communication_turns * self._num_actions))

        self._communication_states = np.zeros((
            self._num_communication_turns, self._num_agents, self._state_size))
        self._communication_states[0] = np.copy(agent_states)
        self._communication_actions = np.zeros(
            (self._num_communication_turns, self._num_agents))

        for comm_turn in xrange(self._num_communication_turns):
            # communication_actions_curr_turn stores the output actions for the current turn.
            communication_actions_curr_turn = np.zeros(self._num_agents)
            # avg_comm_vector is the averaged communication output of all agents.
            avg_comm_vector = np.zeros(self._num_actions)

            # Sample communication actions for all agents.
            # Store each communication action in the communication_actions_curr_turn array.
            # Compute the averaged communication vector in avg_comm_vector.
            for i in xrange(self._num_agents):
                if eval:
                    communication_actions_curr_turn[i] = self._agents[i].best_action(
                        self._communication_states[comm_turn][i])
                else:
                    communication_actions_curr_turn[i] = self._agents[i].sample(
                        self._communication_states[comm_turn][i])
                avg_comm_vector[i][communication_actions_curr_turn[i]] += 1.0
            avg_comm_vector /= self._num_agents
            self._communication_actions[comm_turn] = np.copy(
                communication_actions_curr_turn)

            # Construct the next intermediate state for each agent.
            next_turn_agent_states = []
            for i in xrange(self._num_agents):
                agent_state = np.copy(agent_states[i])
                # Set the appropriate subsection of the agent state to be equal 
                # to the averaged communication output from all agents.
                agent_state[(self._base_state_size + comm_turn * self._num_actions):(
                    self._base_state_size + (comm_turn + 1) * self._num_actions)] = avg_comm_vector
                
                next_turn_agent_states.append(agent_state)
            
            if comm_turn < self._num_communication_turns - 1:
                next_turn_agent_states = np.array(next_turn_agent_states)
                self._communication_states[comm_turn + 1] = np.copy(next_turn_agent_states)

        return self._communication_actions[-1]

    def best_action(self, environment_state):
        """Returns a greedy set of output actions.
           Args:
            environment_state: the current environment state.
           Returns:
            output_actions: a greedy set of output actions from all agents.
        """
        return self.sample(environment_state)

    def store(self, environment_state, output_actions, 
        environment_next_state, reward, terminal, eval=False):
        """Stores transitions from the most recent set of communication turns and output. 
           Args:
            environment_state: input environment_state.
            output_actions: output actions from the multi-agent RL agent.
            environment_next_state: next environment state.
            reward: reward obtained from environment.
            terminal: whether or not current transition led to terminal state.
            eval: whether current episode is an eval episode.
        """
        if not eval:
            for comm_turn in xrange(self._num_communication_turns):
                for i in xrange(self._num_agents):
                    controller = self._agents[i]

                    curr_state = self._communication_states[comm_turn][i]
                    controller_action = self._communication_actions[comm_turn][i]

                    if comm_turn < self._num_communication_turns - 1:
                        next_state = self._communication_states[comm_turn + 1][i]
                        communication_reward = 0
                        communication_terminal = False
                        controller.store(
                            curr_state, controller_action, next_state, communication_reward, communication_terminal)
                    else:
                        ####TODO: Construct the next controller state using the next environment state.
                        next_state = np.copy(curr_state)
                        controller.store(curr_state, controller_action, next_state, reward, terminal)

    def update(self):
        """Updates all the controllers."""
        for controller in self._agents:
            controller.update()