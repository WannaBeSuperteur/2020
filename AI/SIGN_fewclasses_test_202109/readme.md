**How to execute: refer to ```execute.txt```**

https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

License: CC0 Public Domain

Merge original classes into new classes:
* original class ```0, 1, 2, 3, 4, 5, 6, 7, 8``` -> new class ```0```
* original class ```9, 10, 16, 41, 42``` -> new class ```1```
* original class ```11, 19, 20, 21``` -> new class ```2```
* original class ```12, 13, 15, 32``` -> new class ```3```
* original class ```14, 17, 18``` -> new class ```4```
* original class ```22, 23, 24, 25, 26, 27, 28, 29, 30, 31``` -> new class ```5```
* original class ```33, 34, 35, 36, 37, 38, 39, 40``` -> new class ```6```

SHAP: ```algo_0_shap.py```
* Scott M Lundberg and Su-In Lee, "A Unified Approach to Interpreting Model Predictions", Allen School of Computer Science University of Washington, available online at https://arxiv.org/abs/1705.07874.

LIME: ```algo_1_lime.py```
* Marco Tulio Ribeiro, Sameer Singh et al, "Why should i trust you?: Explaining the predictions of any classifier", University of Washington Seattle, WA 98105, USA, available online at https://arxiv.org/abs/1602.04938.

Grad-CAM: ```algo_2_gradcam2.py```
* Ramprasaath R. Selvaraju, Michael Cogswell et al, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", Georgia Institute of Technology, Atlanta, GA, USA, available online at https://arxiv.org/abs/1610.02391.

XCNN: ```algo_3_XCNN.py```
* Amirhossein Tavanaei, 'Embedded Encoder-Decoder in Convolutional Networks Towards Explainable AI', Arxiv 2020, available online at https://arxiv.org/abs/2007.06712.
