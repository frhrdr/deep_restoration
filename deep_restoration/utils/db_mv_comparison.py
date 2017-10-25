

def subset10_paths(classifier):
    assert classifier in ('alexnet', 'vgg16')
    img_hw = 227 if classifier == 'alexnet' else 224
    dir = '../data/selected/images_resized_{}/'
    []