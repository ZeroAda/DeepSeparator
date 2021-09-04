#!/usr/bin/env python
"""
File Name: Paint ratio
Description: 
Author: Orange Killer
Date: 
Log:
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


if __name__ == '__main__':
    train1v1 = np.loadtxt("1v1CNN-CNN train loss")
    test1v1 = np.loadtxt("1v1CNN-CNN test loss")

    train7v1 = np.loadtxt("7v1CNN-CNN train loss")
    test7v1 = np.loadtxt("7v1CNN-CNN test loss")

    train15v1 = np.loadtxt("15v1CNN-CNN train loss")
    test15v1 = np.loadtxt("15v1CNN-CNN test loss")

    train31v1 = np.loadtxt("31v1CNN-CNN train loss")
    test31v1 = np.loadtxt("31v1CNN-CNN test loss")

    plt.figure(figsize=(12,10))
    a = plt.subplot(111)
    plt.plot(range(1,101), train1v1, '-', label='1:1 train loss',c='deepskyblue',linewidth=2.3)
    plt.plot(range(1,101), test1v1, '.-.', label='1:1 test loss',c='deepskyblue',linewidth=2.3)

    plt.plot(range(1, 101), train7v1, '-', label='7:1 train loss',c='steelblue',linewidth=2.3)
    plt.plot(range(1, 101), test7v1, '.-.', label='7:1 test loss',c='steelblue',linewidth=2.3)

    plt.plot(range(1, 101), train15v1, '-', label='15:1 train loss',c='lightseagreen',linewidth=2.3)
    plt.plot(range(1, 101), test15v1, '.-.', label='15:1 test loss',c='lightseagreen',linewidth=2.3)

    plt.plot(range(1,101), train31v1, '-', label='31:1 train loss',c='palegreen',linewidth=2.3)
    plt.plot(range(1,101), test31v1, '.-.', label='31:1 test loss',c='palegreen',linewidth=2.3)

    yminorLocator = MultipleLocator(0.05)
    a.yaxis.set_minor_locator(yminorLocator)
    xminorLocator = MultipleLocator(5)
    a.xaxis.set_minor_locator(xminorLocator)

    xticks = range(0,102,10)
    plt.xticks(xticks,fontsize=12)

    yticks = np.arange(0, 0.7, 0.05)
    plt.yticks(yticks, fontsize=12)



    plt.xlabel("epoch",fontsize=15)


    plt.legend(fontsize=15)

    plt.savefig("ratio.png",dpi=400)


