# deep_restoration - MSc AI Thesis

### Next Steps:
- adapt code to allow stacking models (done - for now)
- pretty up plotting functions (done - for now)


### Notes
- vgg16/l1_cd/run3 yields near perfect reconstructions. no grey areas, no black spots, hard to tell from the original. this indicates that learning the biases is a problem for some reason.
- learning rate at 0.0003 seems good. 0.001 becomes a bit unstable (as in vgg16/l1_cd/run4)
- two difference images for visualization added, one for relative color (optimally grey) and one for absolute difference (optimally black)

- caffe dosovitsky
- gleichzeitig losses für jedes layer gestapelt
- bis pooling 2 und für alexnet

#### Run Schedule

