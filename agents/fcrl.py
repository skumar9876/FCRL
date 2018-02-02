"""
FCRL implementation as described in "Federated Control with Hierarchical Multi-Agent RL" by Kumar et al. 
This implementation is tailored to the scheduling environment implemented in scheduling_env.py.
@author: Saurabh Kumar
"""

from collections import defaultdict
from dqn import DqnAgent
import numpy as np
import sys


class FederatedControlAgent(object):

    def __init__(self,
                 learning_rates=[0.1, 0.00025],
                 state_sizes=[0, 0],
                 constraints=None,
                 num_constraints=0,
                 num_primitive_actions=0,
                 num_controllers=0,
                 num_controllers_per_subtask=0,
                 num_communication_turns=0,
                 critic_fn=None,
                 controller_subset_fn=None):
        """Initializes a FCRL agent.
           Args:
            learning_rates: learning rates of the meta-controller and controller agents.
            state_sizes: state sizes of the meta-controller and controller agents.
                         State sizes are assumed to be 1-dimensional.
            constraints: array of constraints for the meta-controller, which defines its action space.
            num_constraints: number of actions for the meta-controller.
            num_primitive_actions: number of actions for the controller.
            num_controllers: total number of controllers. 
            num_controllers_per_subtask: the number of controllers that coordinate to complete a given subtask.
            num_communication_turns: the number of turns for which controllers communicate.
            critic_fn: a custom critic function for a particular environment.
            controller_subset_fn: a custom function that returns the next controller subset.
        """
        self._meta_controller_state_size = state_sizes[0]

        self._num_controllers = num_controllers
        # Number of controllers that communicate to complete a subtask.
        self._num_controllers_per_subtask = num_controllers_per_subtask

        # A controller's state size is the input state size (the environment state)
        # + the ordering vector size (num_controllers_per_subtask)
        # + the communication vectors from the communication rounds and output round
        # (num_communication_turns * num_primitive_actions).
        self._controller_state_size = state_sizes[1] 
        self._controller_state_size += self._num_controllers_per_subtask
        self._controller_state_size += num_communication_turns * num_primitive_actions

        self._meta_controller = DqnAgent(
            state_dims=state_sizes[0],
            num_actions=num_constraints,
            learning_rate=learning_rates[0],
            epsilon_end=0.01)

        self._controller = DqnAgent(
            learning_rate=learning_rates[1],
                num_actions=num_primitive_actions,
                state_dims=[self._controller_state_size],
                epsilon_end=0.01)

        self._constraints = constraints
        self._num_constraints = num_constraints
        self._num_primitive_actions = num_primitive_actions
        self._num_communication_turns = num_communication_turns
        self._critic_fn = critic_fn
        self._controller_subset_fn = controller_subset_fn

        self._intrinsic_time_step = 0
        self._episode = 0

        # Book-keeping variables.
        # Keeps track of the current meta-controller state.
        self._meta_controller_state = None
        # Keeps track of the current action selected by the meta-controller.
        self._curr_constraint = None
        # Keeps track of the meta-controller's reward for the current meta-controller time step.
        self._meta_controller_reward = 0

        # Keeps track of the constraints tried for current controller subset.
        self._tried_constraints = self.reset_tried_constraints()
        # Keeps track of controllers who have completed coordination in the current episode.
        self._done_controllers = []

    def reset_tried_constraints(self):
        return np.zeros(self._num_constraints)

    def get_meta_controller_state(self):
        """
        Returns the meta-controller state.
        Concatenatates vector representation of the largest selected primitive action 
        with the tried constraints vector.
        """
        state = np.zeros(self._num_primitive_actions)

        if len(self._selected_primitive_actions):
            selected_primitive_actions = np.array(self._selected_primitive_actions)
            max_primtive_action = np.max(selected_primitive_actions)
            state[max_primtive_action] = 1
        state = np.concatenate((state, np.copy(self._tried_constraints)), axis=0)

        return state

    def get_controller_environment_states(env_state):
        """Returns an array of controller environment states."""
        controller_environment_states = np.split(env_state, self._num_controllers)
        return controller_environment_states

    def get_controller_state(self, env_state, constraint, ordering, comm_turn, communication_vector=None):
        """
        Returns the controller state containing the controller's environment state, 
        constraint, ordering vector, and received communication vectors.

        Args:
            env_state: The environment state for the current controller.
            constraint: The constraint provided to the current controller.
            ordering: The current controller's position vector in the overall ordering.
            communication_vector: communication received from other controllers in the current communication turn.
        """
        controller_state = np.zeros(self._controller_state_size)
        
        # Apply the constraint to the environment state.
        env_state_plus_constraint = np.logical_and(env_state, constraint).astype(int)
        env_state_size = np.size(env_state_plus_constraint)

        controller_state[0:env_state_size] = env_state_plus_constraint
        controller_state[env_state_size:env_state_size_size + self._num_controllers_per_subtask] = ordering

        if comm_turn >= 1:
            controller_state[(env_state_size + self._num_controllers_per_subtask + (comm_turn - 1) * num_primitive_actions):(
                env_state_size + self._num_controllers_per_subtask + comm_turn * num_primitive_actions)] = communication_vector

        return np.copy(controller_state)

    def intrinsic_reward(self, env_states, constraints, orderings, selected_actions):
        """Intrinsically rewards a subset of controllers using the provided critic function."""
        return self._critic_fn(
            controller_states, constraints, orderings, selected_actions)

    def construct_orderings(self):
        orderings = []
        for i in xrange(np.size(self._num_controllers_per_subtask)):
            ordering = np.zeros(self._num_controllers_per_subtask)
            ordering[i] = 1
            orderings.append(ordering)
        return orderings

    def controller_bookkeeping_vars(self):
        """
        Returns initilizations for controller states, actions, communications, and outputs.
        """
        # Keeps track of all the controller states.
        controller_states = np.zeros(
            self._num_communication_turns + 1, self._num_controllers, self._controller_state_size)
        # Keeps track of all controllers' selected actions (communication + output).
        controller_actions = np.zeros(
            self._num_communication_turns, self._num_controllers, 1)
        # List that will contain the output actions.
        output_actions = []

        return controller_states, controller_actions, output_actions

    def sample(self, environment_state, controller_ordering, eval=False):
        """Samples a (possibly incomplete) output set of controller actions.
        
        Args:
         environment_state: The state provided by the environment.
         controller_ordering: the ordering of controllers specified by the environment.
         eval: Whether this is a train / test episode.

        """
        meta_controller_state = self.get_meta_controller_state()
        self._meta_controller_states.append(meta_controller_state)

        # Sample a constraint from the meta-controller.
        if not eval:
            constraint = self._meta_controller.sample(meta_controller_state)
        else:
            constraint = self._meta_controller.best_action(meta_controller_state)

        self._tried_constraints[constraint] = 1
        self._curr_constraint = constraint

        controller_environment_states = self.get_controller_environment_states(
            environment_state)

        controller_subset = self._controller_subset_fn(
            controller_ordering, self._done_controllers)

        orderings = self.construct_orderings()

        controller_states, controller_actions, output_actions = self.controller_bookkeeping_vars()

        # Note: Currently only works when the subsets contain only 2 controllers due to the way 
        # in which communication vectors are appended to the controller states.
        previous_turn_communication_vectors = [None, None]  # The latest communication vectors.
        for comm_turn in xrange(self._num_communication_turns + 1):

            communication_vectors = np.zeros(
                self._num_controllers_per_subtask, self._num_primitive_actions)

            for i in xrange(np.size(controller_subset)):
                ordering = orderings[i]

                # Construct the controller state.
                controller_index = controller_subset[i]
                env_state = controller_environment_states[controller_index]
                prev_comm_vector = previous_turn_communication_vectors[(i+1) % self._num_controllers_per_subtask]
                controller_state = self.get_controller_state(
                    env_state, constraint, ordering, comm_turn, prev_comm_vector)

                controller_states[comm_turn][i] = controller_state

                if not eval:
                    action = self._controller.sample(controller_state)
                else:
                    action = self._controller.best_action(controller_state)

                controller_actions[comm_turn][i] = action

                communication_vector = np.zeros(self._num_primitive_actions)
                communication_vector[action] = 1
                communication_vectors[i] = communication_vector
                previous_turn_communication_vectors[i] = communication_vector

                if comm_turn == self._num_communication_turns - 1:
                    output_actions.append(action)

        # Compute the intrinsic reward that all the controllers in the controller
        # subset receive.
        self._intrinsic_reward = self._critic_fn(
            controller_environment_states, constraint, orderings, output_actions)

        # Store the controller transitions.
        for comm_turn in xrange(self._num_communication_turns):
            for i in xrange(np.size(controller_subset)):
                controller_state = controller_states[comm_turn][i]
                controller_action = controller_actions[comm_turn][i]
                controller_next_state = controller_states[comm_turn + 1][i]
                controller_reward = 0
                controller_terminal = False
                if comm_turn == self._num_communication_turns - 1:
                    controller_reward = intrinsic_reward
                    controller_terminal = True

                self._controller.store(controller_state, controller_action, 
                    controller_reward, controller_next_state, controller_terminal, eval)

        # Reset/Update bookkeeping variables.
        if self._intrinsic_reward:
            for controller in controller_subset:
                self._done_controllers.append(controller)
            self._tried_constraints = self.reset_tried_constraints()

        return output_actions

    def best_action(self, environment_state):
        return self.sample(environment_state, eval=True)

    def store(self, state, output_actions, reward, next_state, terminal, eval=False):
        """Stores the current transition in the meta-controller's replay memory.
           The transition is stored in the replay memory of the controller.
           If the transition culminates in a subgoal's completion or a terminal state, a
           transition for the meta-controller is constructed and stored in its replay buffer.
           Args:
            state: current state
            action: primitive action taken
            reward: reward received from state-action pair
            next_state: next state
            terminal: extrinsic terminal (True or False)
            eval: whether the current episode is a train or eval episode.
        """

        curr_meta_controller_state = self._meta_controller_states()[-1]
        action = self._curr_constraint
        next_meta_controller_state = self.get_meta_controller_state()
        self._meta_controller_reward += reward

        self._meta_controller.store(curr_meta_controller_state, self._curr_constraint, 
            self._meta_controller_reward, next_meta_controller_state, terminal, eval)

        self._meta_controller_state = None

    def update(self):
        self._controller.update()
        # Only update meta-controller right after a meta-controller transition has taken place,
        # which occurs only when either a subgoal has been completed or the agent has reached a
        # terminal state.
        if self._meta_controller_state is None:
            self._meta_controller.update()