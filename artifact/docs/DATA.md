# Data and external assets

No raw third-party datasets, encoder weights, or feature stores are included.
The reported experiments use CIFAR-100/CIFAR-100-C, ImageNet-C, ImageNet-R,
ImageNet-Sketch, ImageNet-V2, and public pretrained encoders. Synthetic studies
generate their inputs from locked configurations.

Acquire datasets and models from their official upstream sources and retain
their original terms. Build feature stores with the scripts under
`scripts/campaign/` and place all generated arrays outside the repository. The
protocol and result records retain checksums needed to authenticate the inputs
used for the reported campaigns.

