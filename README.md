# deep_restoration - MSc AI Thesis

### Next Steps:
- test M&V model on vgg. generally, generate tons of images...
- find a way for loading dosovitskiy's caffe model into tensorflow or extract weights somehow
- add split loss option per module
- conitue training of later vgg layers to actual convergence
- work through and reimplement mahendran & vedaldi's paper (done)
- adapt code to allow stacking models (done - for now)
- pretty up plotting functions (done - for now)
- consider adding parts of curet texture data set to selected images (decided not to - for now)

### Notes
- learning rate at 0.0003 seems good. 0.001 becomes a bit unstable (as in vgg16/l1_cd/run4)
- startin at l4, vgg layers seem to require longer to train. either try upping lr, or train in multiple runs.
- chose bmp over png for 3x faster acces at cost of 150% space
- mahendran and vedaldi reference faulty caffe alexnet with first two lrn and mpool layers switched in order. (see discussion here: https://github.com/BVLC/caffe/issues/296) This should be kept in mind when comparing reconstrutions.
- mahendran and vedaldi seem to rescale their results to the interval 0:1 before plotting. then the results if get are nearly identical. Otherwise they tend to be a lot darker.

#### Run Schedule
