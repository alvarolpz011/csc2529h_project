# csc2529h_project

[ ] Write about exactly the difference between 3dgs and NeRF


[ ] (Shayan's) Look into other 3dgs methods that in some way incorporate the D channel (same kind of dataset  and choose that as a baseline)

[ ] (Shayan's) Find specific dataset (or even two)
* DL3DV
* RealEstate10K
* Tartan Air
* ScanNet

### Notes
* The code for Difix3D is in [this github](https://github.com/nv-tlabs/Difix3D).
* A paper did something similar using depth estimation models: [DepthSplat: Connecting Gaussian Splatting and Depth](https://arxiv.org/pdf/2410.13862). We could check what datasets they've used.

### Potential problems: 
* What if we don't find a dataset that includes Depth information?
* Do we have access to the finetuned model they made for DIFIX or the dataset they used? If we dont have the model they used, we would need to replicate their finetuning, which requires the data they used.
* Could we use depth estimation models instead of Lidar points?
* I can't find the exact Stable Difussion model they used as base, just the newest version in Stability AI's Hugging Face page.