# Neural Radiance Fields

**Neural Radiance Fields** were proposed in [this paper](https://arxiv.org/pdf/2003.08934). It uses a sparse set of input views (like many photos of an object taken from different angles with a smartphone) to get new synthetic views (angles) of the scene. This basically achieves a 3D view of the object capable of "rendering" new angles additional to the original sparse angles.

# 3D Gaussian Splatting
It belongs to the group of Radiance Fields, just as NeRF. It also starts with a set of photos from different angles from a subject. These photos can also be extracted from a video as an image sequence. Using the method called [Structure-from-Motion](https://en.wikipedia.org/wiki/Structure_from_motion), a point cloud is made from said images. Each of these points in the point cloud are turned into a 3D Gaussian. One of it's advantages over NeRF is the speed of reconstruction of the scene.

[This short video](https://www.youtube.com/watch?v=Tnij_xHEnXc) illustrates it well.

[This longer viddeo](https://www.youtube.com/watch?v=VkIJbpdTujE) explains it in more detail.

# DIFIX3D+
It is a pipeline that enhances the 3D reconstruction from 3DGS and NeRF using single-step image diffusion models trained to remove artifacts. In [their paper](https://arxiv.org/pdf/2503.01774), it is used: 
* During reconstruction: render novel (pseudo) views from the current 3D representation → clean them with Difix → “distill” the cleaned images back into the 3D model to progressively improve the 3D representation (progressive 3D updates).
* At inference / render time: run Difix as a real-time neural enhancer on rendered images to remove residual artifacts.

## Why DIFIX3D+?
NeRF and 3DGS can produce artifacts in areas that are underobserved, causing bluriness and geometry inconsistencies. Difussion models can be valueble 2D priors given the amount of data they are trained on. The approach addresses both (a) improving the internal 3D model (via distilled cleaned views) and (b) improving final rendered images (via inference-time enhancement).

## Technicalities of DIFIX3D+
The base model used for DIFIX3D+ is [SD-Turbo](https://arxiv.org/pdf/2311.17042), finetuned in a similar
manner to [Pix2pix-Turbo](https://arxiv.org/pdf/2403.12036). 



Some reviewing material: 
* https://arxiv.org/pdf/2308.04079
* https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
* https://arxiv.org/pdf/2503.01774
* https://www.youtube.com/watch?v=VkIJbpdTujE
* https://www.youtube.com/watch?v=Tnij_xHEnXc
* https://www.youtube.com/watch?v=JuH79E8rdKc


