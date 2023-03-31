# Experimentation Process

Firstly, I tried one convolution layer with a 32 Filter,3by3 convolutional layers,no hidden layer or 
I started initially by adding : a convolution layer with 32 filters and a 3by3 kernel and  I got a mean testing accuracy of 89% which was quite ok.
Then I attempted to tweak my model by changing the 32filter to a 128filter in the first convolution layer and added a
pooling layer with a 3x3 pooling size. The results were better and got an average testing of 92% accuracy. I decided to add  another convolution layer with a 132filter and 4by4
kernel and got only a slight increase of accuracy resulting in 94%. I then added another pooling layer but the mean accuracy still remained the same.
So I decided to remove it since it only made the model less efficient without increasing the accuracy. So, I replaced it with a hidden layer of 512 UNITS and a dropout of about 0.2.
The accuracy was about 96%. I added more convolutional layers, hidden layers with dropouts and pooling layers but the accuracy either went lower or did not change.
I left it like that since it was only making the model less efficient without getting more accurate.
