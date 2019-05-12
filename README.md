# VariationalRecurrentNeuralNetwork

Variational RNN demo on MNIST (treated as sequence data)

Paper (VRNN): ![*A Recurrent Latent Variable Model for Sequential Data*.](https://arxiv.org/pdf/1506.02216)  
Project Report (Disentangled VRNN) ![Disentangling latents in a Variational RNN](https://github.com/Abishekpras/vrnn/blob/master/Disentangling%20latents%20in%20a%20Variational%20RNN.pdf)  
## Run:

To train (Disentangled VRNN): ``` python3 train_dis_vrnn.py ```
(VRNN): ``` python3 train.py```
## Sample Reconstructions:
(Top) Sample content while fixing style
(Bottom) Sample style while fixing content
![After 10/100 epochs](pos_samples.png)

## Disentangled VRNN (experimental) Model Generation
![Image Generation model](dis_vrnn_generation.png)

## Disentangled VRNN Inference
![Image Generation model](dis_vrnn_inference.png)


