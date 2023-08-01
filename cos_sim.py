import torch
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import glob
from tqdm import tqdm
import os
from CLIP import clip
import pandas as pd
from pca_plot import pca_plots
# top3の画像パスを返す
def cos_sim(text,device="mps"):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model = model.eval()

    text_input = clip.tokenize([text]).to(device)
    # features.csvの1~512列目の値を取得
    df_origin = pd.read_csv('features.csv', header=None)

    # 数値以外の値を含む可能性がある列を特定し、それらを数値に変換します。
    # これは数値に変換できない値（NaN）を含む可能性があります。
    df = df_origin.apply(pd.to_numeric, errors='coerce')

    # numpy配列に変換します。この際、NaN値は適当な数値（ここでは0）に置き換えます。
    features = df.values[1:, :512]
    features = np.nan_to_num(features)

    # featuresをテンソルに変換
    features = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        text_feature = model.encode_text(text_input).float()
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
    # text_featureとfeaturesのコサイン類似度を計算
    sim = torch.cosine_similarity(text_feature, features, dim=-1)
    # コサイン類似度が高い順にソート
    top3 = torch.argsort(sim, descending=True)[:3]
    result = [top3[i].item() for i in range(top3.shape[0])]
    result_name = [df_origin.values[i+1, 512] for i in result]
    # top3において、513列目を取得
    # top3_path = pd.read_csv('features.csv', header=None).values[top3, 513]
    pca_plots(text_feature)
    return "/".join(result_name)