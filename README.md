# deep_restoration - MSc AI Thesis

### Next Steps:
- change architecture: each module specifies in and out tensor (optionally either from last module or classifier) and whether reconstruction becomes part of the loss
- enable loading weights to continue training
- create reconstructions for deeper vgg layers and alexnet
- find a way for loading dosovitskiy's caffe model into tensorflow or extract weights somehow
- create resized dataset for runtime speedups (chose bmp over png for 3x faster acces at cost of 150% space)
- adapt code to allow stacking models (done - for now)
- pretty up plotting functions (done - for now)


### Notes
- learning rate at 0.0003 seems good. 0.001 becomes a bit unstable (as in vgg16/l1_cd/run4)
- startin at l4, vgg layers seem to require longer to train. either try upping lr, or train in multiple runs.


#### Run Schedule

