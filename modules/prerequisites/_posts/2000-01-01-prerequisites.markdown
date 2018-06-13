---
title: Prerequisites
---

# Prerequisites
Starting with Deep Learning and Keras and understanding this tutorial is simple.

<div>
After checking that you meet all the prerequisites, you can
<a href="{{site.baseurl}}/modules/tutorial/introduction/">start the tutorial</a>.
</div>

## Software Prerequisites
Typically, you should be able to run this tutorial on any python 3 distribution that
includes numpy, scipy, scikit-learn, matplotlib and Keras (with tensorflow backend).
However, you are strongly advised to use one of the following two pre-bundled python
distributions ([Anaconda](https://anaconda.org/anaconda/python) or
[WinPython](https://sourceforge.net/projects/winpython/files/WinPython_3.6/3.6.5.0/))
that include most required python packages.

### Option 1: Use Anaconda
Anaconda has most of the required packages and is available for Windows, Linux, or Mac OS.
To install Anaconda on your OS, you have to download the respective 
installer from its [downloads page](https://www.anaconda.com/download/)

After installing Anaconda, you have to open the Anaconda Prompt and install Keras by
typing in:

```
conda install keras
```

AFter confirming the changes that are recommended, Keras will be installed.

### Option 2: Use WinPython (Windows only)
WinPython is an even better option as it has also preinstalled Keras. However, it only
works for Windows. If you want to use WinPython you can download the
version 3.6.3.0 of WinPython with QT5, which is available
[here](https://sourceforge.net/projects/winpython/files/WinPython_3.6/3.6.3.0/WinPython-64bit-3.6.3.0Qt5.exe) for 64-bit OSes
and [here](https://sourceforge.net/projects/winpython/files/WinPython_3.6/3.6.3.0/WinPython-32bit-3.6.3.0Qt5.exe/download) for
32-bit OSes.

## Other Prerequisites
In terms of other prerequisites etc., you are advised to have the following:

- some basic knowledge about Machine Learning, and specifically classification, regression, etc.
- some basic understanding of data manipulation in Python (numpy, scikit-learn, matplotlib)
- a nice attitude towards learning, because it's going to be fun!

## IDE Prerequisites
As an IDE, you can use anything you want. I use [Spyder](https://pythonhosted.org/spyder/)
for this, which is also preinstalled in the distributions mentioned above.
Having said this, to correctly run all the exampless in Spyder (with IPython), you may
have to select Tools --> Preferences and inside the IPython console and the Graphics tab
set Backend to Automatic. This is shown in the following screenshot:

![Spyder Option for Graphics Backend]({{site.baseurl}}/img/spyder.png)

<div>
Having checked that you meet all the prerequisites, you can
<a href="{{site.baseurl}}/modules/tutorial/introduction/">start the tutorial</a>.
</div>
