import numpy as np
import random

class EventScheduling(object):
	"""
	Environment for the distributed scheduling problem described in 
	Kumar et al. (https://arxiv.org/abs/1712.08266) 
	"""

	def __init__(self, num_agents=4, num_times=8, max_iterations=10):
		"""Initializes a event scheduling environment.

		   Args:
		   	num_agents: the number of events that need to be scheduled.
		   	num_times: the number of possible times for each event.
		   	max_iterations: max number of tries agent has to output a valid schedule.
		"""
		self.num_agents = num_agents
		self.num_times = num_times
		self.max_iterations = max_iterations

	def reset(self):
		"""Resets the environment.

		   Returns:
		   	database_arr: array containing each controller's database.
		   	ordering: ordering of controllers.
		"""
		self.time_step = 0

		times = np.arange(self.num_times)
		# Generate a possible output ordering of times. 
		possible_ordering = np.sort(random.sample(times, self.num_agents))
		# Generate a controller ordering.
		self.controller_ordering = np.arange(self.num_agents)
		np.random.shuffle(self.controller_ordering)

		# Based on the controller ordering and possible times output,
		# construct the controllers' databases.
		self.databases = np.zeros(self.num_times * self.num_agents)
		self.databases_dict = {}
		for i in xrange(self.num_agents):
			curr_controller = self.controller_ordering[i]

			# Each database is a multi-hot vector indicating the available times
			database = np.zeros(self.num_times)
			database[self.possible_ordering[curr_controller]] = 1

			# Set random subset of entries in the database to 1.
			subset_size = random.randint(0, self.num_times - 1)
			if subset_size > 0:
				all_times = np.arange(self.num_times)
				database[random.sample(all_times, subset_size)] = 1

			self.databases_dict[self.curr_controller] = np.copy(database)
			self.databases[self.num_times * self.curr_controller : self.num_times * (
				curr_controller + 1)] = np.copy(database)

		return np.copy(self.databases), np.copy(self.controller_ordering)

	def step(self, output_schedule):
		self.time_step += 1

		terminal = (self.score(output_schedule) == 1) or (self.time_step >= self.max_iterations)

		return self.score(output_schedule), terminal
	
	def score(self, output_schedule):
		"""Returns the score for a given output schedule. 

			Args:
			 output_schedule: the proposed schedule.

			Returns:
			 score: +1 if schedule is valid, 0 otherwise.
		"""
		if self.time_step >= self.max_iterations:
			return 0

		# Checks whether output schedule is complete or incomplete. 
		# Output schedule is incomplete if its length is not equal to the number of agents.
		# If output schedule is incomplete, returns 0.
		if np.size(output_schedule) < self.num_agents:
			return 0

		# Checks that the output schedule is in strictly increasing order
		# according to the controller ordering and that each output time
		# is in the corresponding controller's database.
		val = -1
		for i in xrange(len(output_schedule)):
			controller_index = self.controller_ordering[i]

			if val >= output_schedule[controller_index]:
				return 0

			elif not self.databases[controller_index][output_schedule[controller_index]]:
				return 0

			val = output_schedule[controller_index]

		return 1

def critic_function(databases, constraints, orderings, selected_times):
	"""Provides intrinsic reward for subset of controller output times - used for FCRL agent.

	   Args:
	   	databases: The controllers' databases.
	   	contraints: The constraints selected by the meta-controller.
	   	orderings: The position vectors of the controllers.
	   	selected_times: The times selected by the controllers.

	   Returns:
	   	1 if selected times satsify both the databases & constraints and are in the correct order,
	   	0 otherwise.
	"""

    # Checks that each selected time is in the corresponding controller's database plus constraint.
    for i in xrange(len(databases)):
        database_plus_constraint = np.logical_and(databases[i], constraints[i]).astype(int)
        if not database_plus_constraint[selected_times[i]]:
            return 0

    controller_ordering = np.zeros(len(orderings))
    # Checks that the times are in strictly increasing order with respect to the controller ordering.
    for i in xrange(len(orderings)):
    	ordering = orderings[i]
    	controller_index = ordering.index(1)
    	controller_ordering[controller_index] = i

    val = -1
    for controller_index in xrange(len(controller_ordering)):
    	if val >= selected_times[controller_ordering[controller_index]]:
    		return 0
    	val = selected_times[controller_ordering[controller_index]]

    return 1

def get_next_controller_pair(controller_ordering, done_controllers):
	"""
	Returns the next controller pair given a list of completed controllers.

	Args:
	 controller_ordering: The ordering of all the controllers.
	 done_controllers: The controllers which have been used so far.

	Returns:
	 controller_pair: Array of indices of the next two controllers.
	"""
	last_controller = done_controllers[-1]
	last_controller_index = controller_ordering.index(last_controller)

	return controller_ordering[last_controller_index + 1:last_controller_index + 3]
