import model.svhn2mnist
import model.usps
import model.syn2gtrsb

def Generator(source, target, pixelda=False):
    if source == 'usps' or target == 'usps':
        return model.usps.Feature()
    elif source == 'svhn':
        return model.svhn2mnist.Feature()
    elif source == 'synth':
        return model.syn2gtrsb.Feature()

def Classifier(source, target):
    if source == 'usps' or target == 'usps':
        return model.usps.Predictor()
    if source == 'svhn':
        return model.svhn2mnist.Predictor()
    elif source == 'synth':
        return model.syn2gtrsb.Predictor()
