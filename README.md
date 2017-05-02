# deep_restoration - MSc AI Thesis

### Next Steps:
- test deconv-relu-deconv and deconv-relu-conv models
- find second source on image net means
- vary learning rate
- train and stack 3 layer inversions, compare to same model trained in one go.
- pretty up plotting functions
- redo vgg experiments with BGR mean order (done)
- test artificial data with uniform areas to test vggnet (done)
- track source of black spots, try renormalizing again (done)
- rework run logging, so used paramters can be read off (done)
- unify vvg and alexnet layer inversion classes (done)
- track validation set loss (done)
- find good way to log loss per channel (done)

### Run Keys:

#### vggnet
- run3: 1x1 conv 3x3 deconv
- run4: 3x3 conv 3x3 deconv
- run5: 5x5 conv 5x5 deconv
- run6: 7x7 conv 7x7 deconv

#### alexnet
- run1: 5x5 conv 5x5 deconv
- run2: 11x11 conv 11x11 deconv

### Notes


#### Run Schedule
- rerun 5x5 l1_cd on vgg with 3x learning rate, see if black spots persist: they do
- then rerun with norm bgr as target (adapt visualization accordingly):
vgg16/l1_cd/run3 yields near perfect reconstructions. no grey areas, no black spots.