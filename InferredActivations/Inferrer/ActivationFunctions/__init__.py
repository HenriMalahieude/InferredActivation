from .Sigmoid import *
from .Tanh import *
from .Gelu import *
from .DoubleReLU import *
from .PiLu import *
from .NewSigmoid import *
from .NewNewSigmoid import *

#Let's try a wrapper.... 
#It ended up not being a wrapper really...
def ApproximatorMode(self, mode, init, apply, extract, *args, **kwargs):
	if mode == 'init':
		return init(self, *args, **kwargs)
	elif mode == 'apply':
		return apply(self, *args, **kwargs)
	else:
		return extract(self, *args, **kwargs)
	
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

#def NewSigApproximator(self, mode, *args, **kwargs):
	#return ApproximatorMode(self, mode, NSInit, NSApply, NSExtract, *args, **kwargs)

def NewNewSigApproximator(self, mode, *args, **kwargs):
	return ApproximatorMode(self, mode, NNSigInit, NNSigApply, NNSigExtract, *args, **kwargs)