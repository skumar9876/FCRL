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

		   Returns:
		    databases: dictionary containing each controller's database.
		"""
		self.num_agents = num_agents
		self.num_times = num_times

		times = np.arange(self.num_times)
		# Generate a possible output ordering. 
		self.possible_ordering = np.sort(random.sample(times, self.num_agents))
		# Based on this ordering, construct the controllers' databases.

		self.databases = {}
		for i in xrange(self.num_agents):
			# Each database is a multi-hot vector indicating the available times
			database = np.zeros(self.num_times)
			database[self.possible_ordering[i]] = 1

			# Set random subset of entries in the database to 1.
			subset_size = random.randint(0, self.num_times - 1)
			if subset_size > 0:
				all_times = np.arange(self.num_times)
				database[random.sample(all_times, subset_size)] = 1

			self.databases[i] = np.copy(database)

	def get_databases(self):
		return self.databases
 
	def score(output_schedule):
		"""Returns the score for a given output schedule. 

			Args:
			 output_schedule: the proposed schedule.

			Returns:
			 score: +1 if schedule is valid, -1 if invalid.
		"""

		# Checks that the output schedule is in strictly increasing order 
		# and that each output time is in the corresponding controller's
		# databse.
		for i in xrange(len(output_schedule)):
			val = output_schedule[i]
			if val > output_schedule[i] or (
				val >= 1 and val >= output_schedule[i]):
				return -1
			elif not self.databases[i][val]:
					return -1
		return 1
