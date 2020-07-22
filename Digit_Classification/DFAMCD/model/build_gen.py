import model.svhn2mnist
import model.usps

def Generator(source, target, pixelda=False):
    if source == 'usps' or target == 'usps':
        return model.usps.Feature()
    elif source == 'svhn':
        return model.svhn2mnist.Feature()
    #elif source == 'synth':
        #return syn2gtrsb.Feature()

def Classifier(source, target):
    if source == 'usps' or target == 'usps':
        return model.usps.Predictor()
    if source == 'svhn':
        return model.svhn2mnist.Predictor()
    
#def ResnetBlock():
#    return model.svhn2mnist.ResnetBlock()
