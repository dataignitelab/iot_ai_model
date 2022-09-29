import timm

def inceptionv4(num_classes = 1):
    model = timm.create_model('inception_v4', pretrained=False, num_classes = 1)

    return model