# deep_restoration - MSc AI Thesis

### Next Steps:
- unify vvg and alexnet layer inversion classes (done)
- rework run logging, so used paramters can be read off (dump params into file)
- redo vgg experiments with BGR mean order
- find good way to log loss per channel (done)
- try deconv-relu-deconv and deconv-relu-conv models
- test artificial data with uniform areas to test vggnet
- find second source on image net means
- train and stack 3 layer inversions, compare to same model trained in one go.
- track validation set loss
- pretty up plotting functions

### Run Keys:

#### vggnet
- run3: 1x1 conv 3x3 deconv
- run4: 3x3 conv 3x3 deconv
- run5: 5x5 conv 5x5 deconv
- run6: 7x7 conv 7x7 deconv


#### alexnet
- run1: 5x5 conv 5x5 deconv
- run2: 11x11 conv 11x11 deconv

#### Notes

VGG v AlexNet differences
+ VGG takes images in range 0 to 1, AlexNet in range 0 to 255
+ VGG normalizes with channel wise mean of the training set, AlexNet with global mean
+ layer dimensions and names are obsiously different
+ AlexNet has strided convolutions and local response normalization
