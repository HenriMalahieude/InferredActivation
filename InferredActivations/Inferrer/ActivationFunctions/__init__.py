from .Sigmoid import SigApply, SigExtract, SigInit, SigBoundaryApply
from .Tanh import TanhApply, TanhExtract, TanhInit
from .Gelu import GeluApply, GeluExtract, GeluInit
from .DoubleReLU import DoubleReLUApply, DoubleReLUExtract, DoubleReLUInit
from .PiLu import PiLUApply, PiLUExtract, PiLUInit

#Let's try a wrapper.... 
#It ended up not being a wrapper really...
def ApproximatorMode(self, mode, func1, func2, func3, *args, **kwargs):
	if mode == 'init':
		return func1(self, *args, **kwargs)
	elif mode == 'apply':
		return func2(self, *args, **kwargs)
	else:
		return func3(self, *args, **kwargs)
	
def SigApproximator(self, mode, *args, **kwargs):
	return ApproximatorMode(self, mode, SigInit, SigApply, SigExtract, *args, **kwargs)

#NOTE: This seems to cause "WARNING:tensorflow:Gradients do not exist for variables ... when minimizing the loss."
def SigBoundaryOnlyApproximator(self, mode, *args, **kwargs):
	return ApproximatorMode(self, mode, SigInit, SigBoundaryApply, SigExtract, *args, **kwargs)

def TanhApproximator(self, mode, *args, **kwargs):
	return ApproximatorMode(self, mode, TanhInit, TanhApply, TanhExtract, *args, **kwargs)

def GeluApproximator(self, mode, *args, **kwargs):
	return ApproximatorMode(self, mode, GeluInit, GeluApply, GeluExtract, *args, **kwargs)

def DoubleReLUApproximator(self, mode, *args, **kwargs):
	return ApproximatorMode(self, mode, DoubleReLUInit, DoubleReLUApply, DoubleReLUExtract, *args, **kwargs)

def PiLuApproximator(self, mode, *args, **kwargs):
	return ApproximatorMode(self, mode, PiLUInit, PiLUApply, PiLUExtract, *args, **kwargs)