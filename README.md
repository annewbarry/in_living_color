### In Living Color: Black and White Photo Colorization Using Neural Networks
## by Anne Barry

**Background and Prior Research**

![My father (second from left) and his siblings on Easter, circa 1952](/img/barry_kids.jpg)
I've always been interested in history, both in broad and personal contexts.  When I visit my parents, I spend hours looking through old photographs, trying to imagine what life was like for my family in different eras.  I am fascinated by colorized versions of historical photos. Historians and artists posit that colorizing photos from eras usually seen only in grayscale [activate our visual memory and evoke empathy](https://www.atlasobscura.com/articles/colorized-historical-photos) and I began to wonder how technology plays a role in image colorization. Some of the best resutls still come from doing it by hand through motnhs of meticulous research and work. Much progress has been made in the domain of using convolutional neural networks and one of the most frequently referenced works on this topic is [Colorful Image Colorization](http://richzhang.github.io/colorization/) by Zhang, Isola, and Efros.  Their work produced stunning results in some categories, especially in the case of nature scenes, but less than perfect results for situations where the color choice is much more ambiguous.


By training a model using a gray version of an existing color photo with the target being the known colorization, Zhang's model was very good at predicting colors for shapes and subjects that are are commonly associate with certain colors.  The two photos below illustrate this point.  Grass is green and there's a good chance that a multicolored dog is white and brown, so their model was successful.  The Yoda mural shines light one of the main challenges with colorization. When a photo does not provide context for a color (such as the seemingly random choice of paint for Yoda's body), the model defaults to using brown, which is a conservative coice. For example, incorrectly choosing highly saturated green over highly saturated red is more costly than choosing brown over highly saturated red because red is further from green in pixel value than it is from brown. In this case though, we probably wouldn't have minded seeing the wrong color for Yoda just as long as we saw some bright color.  Many models that came before Zhang used mean squared error as their loss function which incentivizes this conservative choice.  Zhang et al's loss function was built on categorical cross entropy as well as the probability distribution of the pixels in the training set and showed great gains over using just mean squared error, though as one can see in this example, avoiding brown tones altogether can be quite difficult!

![](/img/zhang_results.png)


Another innovation that Zhang et all made was their use of the Lab color space instead of the RGB color space for the sake of dimensionality reduction.  A photo in the RGB colorspace has three layers (red, green, blue) and each pixel in each layer has a value in the range 0-255.  Training a neural network with a 3 layer grayscale input and a 3 layer RGB target is computationally expensive.  A photo in the Lab color space also has three layers but they are L (lightness, essentially a grayscale), a (red-green), and b (yellow-blue). Therefore, to train a model, you can use the L layer as the input and train with the additional two layers as the target, saving a lot of computational power, and then combine all three layers for the final colorized photo.

![](/img/lab_rbg.png)

In this project, I explore the tradeoff between the traditional L2/mean squared error loss function and customized loss functions I've written and the results they produce in colorizing photos. To do this, I built a convolutional neural network with a custom image data generator.

**Data & Technology**

I used two datasets for photos: the [celebrity dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) from Liu, Luo, Wang, and Tang at the Chinese University of Hong Kong and the [natural images](https://www.kaggle.com/prasunroy/natural-images) dataset from Kaggle.com.  The celebrity dataset has over 200,000 photos of celebrities in front of a wide variety of backgrounds, wearing many types of clothing and in standing in different poses.  I chose to use celebrities because Zhang's research showed that while skin and hair color might be predictable enough for a network to colorize, clothing and other man made objects in photos of people are very hard to colorize with precision because of the lack of context.  Within the natural images dataset, I used the landscapes subset (4319 photos) because the colors of landscapes should be easier to predict.  During this project, I used the below technologies.

![](/img/tech_stack.png)

**Building the Model**
I chose this project both because of my interest as well as my desire to learn more about convolutional neural networks. Ultimately, my goal was to build a simplified version of Zhang's model so that I could then explore the role of the loss function and context.  Going in, I knew that there was a good chance that simply building the CNN would be the biggest challenge, and I was right!  Major challenges included building a custom image data generator & processors well as making the model work with the dimensions of my data.

To process images, I loaded them from the directory as RGB images and resized them to ```(256, 256, 3)``` and normalized the values by multiplying by 1/255.  Then the images were converted to Lab space and further divided by 128, as the pixel values for the ab layers in Lab space range from -128 to +128.  Then the L layer was made into the features and the top two layers were the target.  Images were processed one at a time and placed into batches by the image data generator. The code for the image data generator was adapted from [Custom Data Generators](https://towardsdatascience.com/writing-custom-keras-generators-fe815d992c5a).

The Keras neural network I used was based on the framework developed by [Emil Wallner](https://www.floydhub.com/emilwallner/projects/color/43/code/Alpha-version/alpha_version.ipynb).  His model contained 9 convolutional layers and 3 upsampling layers.  The convolutional layers all used Relu activation with the exception of the last one which used hyperbolic tangent so that the pixel values would be between -1 and 1.  I tested out tweaks of his model, changing all activation functions to tanh and also added dense layers, which made it challenging to make sure I got the proper dimensions for my output ```(batch_size, 256, 256, 2)```.  My model had 10 convolutional layers, 4 upsampling layers, 2 dense layers, and 1 dropout layer.

For evaluation metrics, I paid attention to accuracy but focused on plausability as that is ultimately the goal.  If the shades of green of grass are slightly inaccurate but interally logically consistent, I would prefer that over something more accurate but less believable.  For this reason, my evaluation at this stage is highly subjective.  I tried out many different loss functions and models and once I feel that something is tending towards a correct colorization, I will start to use more formal evaluating metrics.


The standard loss function people in this space have used is mean squared error.  However, this incentivizes the model to play it safe in uncertain situations (ie: Yoda!).  If the true color is highly saturated green (let's say -70 for a pixel on the "a" layer) but thinks it might be a red (+70), the penalty for incorrectly choosing red is 4 times as great as that for incorrectly choosing an even mix between the two (0), even though it might have looked completely realistic if the mural was red when it should have been green.  Therefore, my goal was to create a loss function that balanced incentivizing accuracy (as MSE already does) with penalizing safe choices (ie: pixel values close to 0 that end up creating brown/sepia tones).  In the ```functions.py``` file you will find several that I tried but the one below is the one that worked the best.  I toyed with the hyperparameters around the MSE term as well as my additional term.  The custom loss function at this point only accomplishes the goal of punishing neutral results, but requires much more tuning and development as this project continues to truly get robust colorization.

![](/img/loss_function.png)

**Testing and Results**

Differences in context:

When the model was trained using celebrities using a mean squared error, the resulting predictions were nearly completely tinted with a orange-brown color, which was to be expected, as the photos of celebrities are unpredictable in colorziation and the MSE loss incentivizes conservative color choices in the face of uncertainty.  It predicted poorly on test photos of celebrities as well as landscapes.

![Results from training on celebrities using MSE](/img/celebrities_mse.png)


Differences in error function:
I decided to focus on the landscapes, knowing that I would likely be able to get to some sort of colorization sooner than I would with the celebrities.  I trained the dataset on landscape photos using mean squared error as well as my custom loss function.  One can see in the results that my custom loss function resulted in a bit more colorization.  The histogram backs that up, showing that the model using the custom loss function had a wider distribution of pixel values for the farmscape than the mean squared error model did.

![](/img/loss_function_results.png)

In looking at my results from my custom loss funciton, I noticed most photos had blue and yellow tints but very little green or red.  The figure below shows that pixel ranges for the "b" (blue - yellow) layer of this model's outputs were much greater than that of the "a" (red-green) layer.  Upon further investigation, it looks like the original distributions for the red-green and blue-yellow layers of the farmhouse photo were reasonably similar. However, both the mean squared error and custom loss function disproportionately compressed the red-green values, making those colors virtually indetectable.  One thought as to why this happens is that the contexts in which yellow and blue appear may be more complementary than the contexts in when red and green appear, meaning that the model could "know" that green or red should appear, but can't tell if the leaf is green or a part of brilliant foliage so it defaults to a pixel value of 0.  This, however, requires further research.

![](/img/farmhouse_pixels.png)

**Conclusion and Next Steps**

In this project I accomplished my goal of learning more about neural networks and the role of context and loss funciton in image colorization.  Because I spent the bulk of my time building a working model, I didn't get to devote as much time as I would have liked to developing a more sophisticated loss function.  Additionally, I would like to build a model that uses RGB images and see how the results compare.  Lastly, I want to explore strategies for colorizing old photos, like the one of my dad and his siblings. Both my model and Zhang's model don't quite do the Barry siblings justice, most likely due the quality of the photo.  Perhaps adding in some more image preprocessing (in particular a Gaussian blur) for the training set as well using an autoencoder to denoise my family photos would be helpful in that pursuit.

![](/img/barry_colorized.png)


