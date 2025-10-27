# technical_challenge_AI

## Image Generation Project with Diffusion Model

### Description
This project aims to develop images from scratch using a simple idea from the concept of diffusion models. This type of model was chosen because, as presented in the article [Diffusion Models Beat GANs on Image Synthesis
](https://arxiv.org/pdf/2105.05233), they can achieve superior sample quality when compared to state-of-the-art GANs, especially in tasks involving the generation of high-quality images and audio.

Diffusion models work through a process of degradation and reconstruction. First, the training data is gradually corrupted by adding Gaussian noise, removing details until the original information is practically lost. Next, a neural network is trained to reverse this process, learning to remove the noise until it reconstructs a clean and coherent final sample.

### Frameworks and Libraries Used
- TensorFlow  
- Keras  
- Matplotlib  
- NumPy  
- Scikit-learn  

### Instructions for Training and Image Generation
1. Run the entire `train_script` notebook to train the model. At the end, the model will be saved automatically.  
2. Load the saved model into the `inference_script` notebook and run it.  

### Evaluation Metrics
The metrics used to evaluate image quality follow the reference [in this article](https://www.scirp.org/journal/paperinformation?paperid=90911):
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)  
- **PSNR** (Peak Signal-to-Noise Ratio)

### Results Obtained
- The results of the image quality assessment were: MSE = 0.0069, RMSE = 0.0831, and PSNR = 21.92 dB. These values indicate low reconstruction error (MSE and RMSE close to zero) and acceptable image quality, within the reference range for PSNR (approximately 20–25 dB). This demonstrates that the model was effective for the proposed problem, which consisted of applying noise to images from the MNIST dataset and training a neural network to learn the noise removal pattern, generating clean, high-quality images.
