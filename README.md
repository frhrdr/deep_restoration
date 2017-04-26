# deep_restoration - MSc AI Thesis

### Next Steps:
- train and stack 3 layer inversions, compare to same model trained in one go.
- find good way to log loss per channel
- try deconv-relu-deconv and deconv-relu-conv models
- double check image net means
- make 2k validation split
- pretty up plotting functions
- test artificial with uniform areas data to test vggnet
- rework run logging, so parameters are easily accessible

### run key

#### vggnet
run3: 1x1 conv 3x3 deconv
run4: 3x3 conv 3x3 deconv
run5: 5x5 conv 5x5 deconv
run6: 7x7 conv 7x7 deconv


#### alexnet
run1: 5x5 conv 5x5 deconv
run2: 11x11 conv 11x11 deconv

