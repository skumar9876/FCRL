import numpy as np
import random

class EventScheduling(object):
	"""
	Environment for the distributed scheduling problem described in 
	Kumar et al. (https://arxiv.org/abs/1712.08266) 
	"""

	def __init__(self, num_agents=4, num_times=8):
		"""Initializes a event scheduling environment.

		   Args:
		   	num_agents: the number of events that need to be scheduled.
		   	num_times: the number of possible times for each event.
		"""
		self.num_agents = num_agents
		self.num_times = num_times

	def reset(self):
		"""Resets the environment.

		   Returns:
		   	database_arr: array containing each controller's database.
		   	ordering: ordering of controllers.
		"""
		times = np.arange(self.num_times)
		# Generate a possible output ordering of times. 
		possible_ordering = np.sort(random.sample(times, self.num_agents))
		# Generate a controller ordering.
		self.controller_ordering = np.arange(self._num_agents)
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
	
	def score(self, output_schedule):
		"""Returns the score for a given output schedule. 

			Args:
			 output_schedule: the proposed schedule.

			Returns:
			 score: +1 if schedule is valid, -1 if invalid.
		"""

		# Checks whether output schedule is complete or incomplete. 
		# Output schedule is incomplete if its length is not equal to the number of agents.
		# If output schedule is incomplete, returns 0.
		if np.size(output_schedule) < self.num_agents:
			return 0

		# Checks that the output schedule is in strictly increasing order
		# according to the controller ordering and that each output time
		# is in the corresponding controller's database.
		val = output_schedule[self.controller_ordering[0]]

		for i in xrange(len(output_schedule)):
			controller_index = self.controller_ordering[i]

			if val > output_schedule[controller_index] or (
				i >= 1 and val >= output_schedule[controller_index]):
				return 0

			elif not self.databases[controller_index][output_schedule[controller_index]]:
					return 0

			val = output_schedule[controller_index]
		return 1
