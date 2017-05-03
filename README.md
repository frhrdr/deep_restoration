# deep_restoration - MSc AI Thesis

### Next Steps:
- compile results, link latex doc to repo
- train and stack 3 layer inversions, compare to same model trained in one go.
- find second source on image net means
- step to later layers, for simplicity with vgg first. (done)
- adapt code to allow stacking models (done - for now)
- evaluate which model (cd, dc, dd) works best on alexnet and vgg (done)
- reproduce cd runs on alexnet (done)
- vary learning rate (done)
- redo vgg experiments with BGR mean order (done)
- test artificial data with uniform areas to test vggnet (done)
- track source of black spots, try renormalizing again (done)
- rework run logging, so used parameters can be read off (done)
- unify vvg and alexnet layer inversion classes (done)
- track validation set loss (done)
- find good way to log loss per channel (done)
- pretty up plotting functions (done - for now)

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
- vgg16/l1_cd/run3 yields near perfect reconstructions. no grey areas, no black spots, hard to tell from the original. this indicates that learning the biases is a problem for some reason.
- learning rate at 0.0003 seems good. 0.001 becomes a bit unstable (as in vgg16/l1_cd/run4)
- two difference images for visualization added, one for relative color (optimally grey) and one for absolute difference (optimally black)


#### Run Schedule
- vgg pool1 to conv1_2/relu model as dc - stack to see if dccdcd is better than cdcdcd
- stacked dccdcd pool1 to bgr_normed model - see if results are worse than when layerwise trained
- further layers in alexnet

