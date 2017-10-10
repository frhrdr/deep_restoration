# deep_restoration - MSc AI Thesis

### Notes
- learning rate at 0.0003 seems good. 0.001 becomes a bit unstable (as in vgg16/l1_cd/run4)
- starting at l4, vgg layers seem to require longer to train. either try upping lr, or train in multiple runs.
- chose bmp over png for 3x faster acces at cost of 150% space
- mahendran and vedaldi reference faulty caffe alexnet with first two lrn and mpool layers switched in order. (see discussion here: https://github.com/BVLC/caffe/issues/296) This should be kept in mind when comparing reconstrutions.
- mahendran and vedaldi seem to rescale their results to the interval 0:1 before plotting. then the results if get are nearly identical. Otherwise they tend to be a lot darker. Evidence for this can be found in their code: aravindhm/deep-goggle/core/invert_nn.m in lines 189-191 uses vl_imsc and vl_imarraysc, both of which rescale the image.
- mv on alexnet: exact repoduction (except lrn/pool order)
- mv on vgg16: l1 - l18 conv/relu and pool layers up to poo5 l19+20 are fc/relu and l21 is fc8/lin. lambdaTV is 0.5 for conv1 and 2, 5.0 for conv3 and 4 and 50 for conv5 and fc.
- mv on vgg16 also had some issues with overshooting gradients, depending on initialization. with gradient clipping and several tries, results ware odtained for all layers.
