import tensorflow as tf

def lr_schedule_creator(epoch_sched, ratio):
	def learn_rate_scheduler(epoch, lr):
			interval_check = epoch % epoch_sched
			if interval_check == 0:
				return lr * min(max(ratio, 0), 1)
			return lr
	
	return learn_rate_scheduler