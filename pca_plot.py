import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import glob
import sklearn.decomposition as PCA
import sys
import torch
import japanize_matplotlib
matplotlib.use('Agg')
def pca_plots(txt_vec):
    df = pd.read_csv("features.csv")
    pca = PCA.PCA(n_components=2)
    z = txt_vec.tolist()[0]
    print(z)
    z.append("a")
    newdf = pd.DataFrame([z],columns=df.columns)
    df = pd.concat([df,newdf],ignore_index=True)
    data_array = pca.fit_transform(df.iloc[:,:512])
    for i in range(len(data_array)):
        if i < len(data_array)-1:  # 最後の行でない場合
            if i == 0:
                plt.scatter(data_array[i, 0], data_array[i, 1], color='green', label='1st Img')
            elif i == 1:
                plt.scatter(data_array[i, 0], data_array[i, 1], color='orange', label='2nd Img')
            elif i == 2:
                plt.scatter(data_array[i, 0], data_array[i, 1], color='blue', label='3rd Img')
            else:
                plt.scatter(data_array[i, 0], data_array[i, 1], color='black', label='Other Img')
        else:  # 最後の行の場合
            plt.scatter(data_array[i, 0], data_array[i, 1], color='red', label='Your text')
    plt.legend()
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig("./public/plot.png")
    plt.close()
    return