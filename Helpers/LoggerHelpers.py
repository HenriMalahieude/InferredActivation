import logging

def create_logger(filename):
	logger = logging.getLogger("internal_logger")
	logger.setLevel(logging.DEBUG)
	fileHandle = logging.FileHandler(filename=filename)

	formatter = logging.Formatter(fmt='%(message)s')
	fileHandle.setFormatter(formatter)
	logger.addHandler(fileHandle)
	return logger

def output_training_history(logger, hist):
	logger.info("\nTraining Loss: {}".format(hist.history["loss"]))
	logger.info("Training T5: {}".format(hist.history["T5"]))
	logger.info("Training T3: {}".format(hist.history["T3"]))
	logger.info("Training T1: {}".format(hist.history["T1"]))

def output_validation_history(logger, hist):
	logger.info("\nValidate Loss: {}".format(hist.history["val_loss"]))
	logger.info("Validate T5: {}".format(hist.history["val_T5"]))
	logger.info("Validate T3: {}".format(hist.history["val_T3"]))
	logger.info("Validate T1: {}".format(hist.history["val_T1"]))

def output_custom_history(logger, hist, accessor):
	logger.info(f"{accessor}: {hist.history[accessor]}")
	