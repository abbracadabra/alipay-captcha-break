**Accuracy is horrible,about 20% i guess.**
***
This is the captcha we are trying to bypass:[link](https://authsu18.alipay.com/login/index.htm)

<div align='left'>
  <img src='https://user-images.githubusercontent.com/35487258/50561572-fe7d3100-0d46-11e9-9a7c-e780524e9626.png' height="225px">
</div>
...
scripts includes:
- gensinglecontour.py #generate images for training
- trainsinglecontour.py #training
- predictcontour.py #predict
...
To predict,run ```python predictcontour.py```.Edit line 17 to change the image location.

