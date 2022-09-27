from tensorflow.keras.applications import resnet50 as resnet

def resnet50():
    model = resnet.ResNet50(weights=None, classes = 1, classifier_activation='sigmoid')
    return model