# Face-Aging-GAN
In this repository, we provide a Generative Model to adress the **Face Aging** problematic. We want to translate the input image to the desired class (10-19, 20-29, 30-39, 40-49, 50+) of age given as input too, the firgures bellow showcase what we want.

![Output image](./for_readme/img/input_img.PNG "Input image (256x256)")
![Arrow image](./for_readme/img/arrow.jpg)
![Input image](./for_readme/img/output_img_class_5.PNG "Output image (256x256) translated to the class 5 (age 50+)")

To do so, we train two networks in a competitive manner based on the **Generative Adversarial Nets** framework presented by **Ian Goodfellow** in 2014. The **Generator** (G) and the **Discriminator** (D) gets to see the same input, the difference is in the output. G gives an 256x256 image while D outputs a score corresponding to the probability of the image of being real.

In order to keep the same identity we add an Identity preservation module. We use a Face-Recognition model to get an embedding vector of the input and output images, we thus compute the cosine-similarity between the two and penalize the G network accordingly.

The same idea is used for the age, i.e we penalize the network when the **age classifier** (network) outputs a class that is different from the desired one. In fact we train on 5 classes of ages:
* 0: Ages from 10-19
* 1: Ages from 20-29
* 2: Ages from 30-39
* 3: Ages from 40-49
* 4: Ages 50+

We also made use of the perceptual loss that is computed with a pretrained **VGG_19** and finaly an L2 loss.

P.S: feel free to comment and contribute to this modest work ;-)

# TO-DOs
* ~~Implement age loss~~
* ~~Implement DataParralel~~
* ~~Use multithreads with dataloader~~
* Train a face age classifier (on CACD dataset?) and use it as pretrained in this project.