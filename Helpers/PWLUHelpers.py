def PWLU_Set_StatisticalToggle(model, layer_internal_name = None, to = None):
	assert type(layer_internal_name).__name__ == "str" or type(layer_internal_name).__name__ == "NoneType"

	for i in range(len(model.layers)):
		lyr_name = type(model.layers[i]).__name__
		if lyr_name == "PiecewiseLinearUnitV1":
			model.layers[i].StatisticalAnalysisToggle(to)
		elif layer_internal_name != None and lyr_name == layer_internal_name:
			if hasattr(model.layers[i], "StatisticalAnalysisToggle") and callable(model.layers[i].StatisticalAnalysisToggle):
				model.layers[i].StatisticalAnalysisToggle(to)
			else:
				raise AttributeError(f"No Statistical Analysis Toggle found for: {layer_internal_name}")
			
	print(f"\tFinished Setting PWLU internal Toggle to {to}")
