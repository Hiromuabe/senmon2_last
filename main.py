from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import glob
from tqdm import tqdm
import os
from CLIP import clip
import io
from cos_sim import cos_sim
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import Optional
import shutil
import os
import japanize_matplotlib
from pca_plot import pca_plots


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search/")
def search_image(request: Request, query: str = Form(...)):
    image_paths = cos_sim(query) # 'a.jpg/b.jpg/c.jpg' という形式を仮定
    image_list = image_paths.split('/')  # 画像パスをリストに分割
    return templates.TemplateResponse("index.html", {"request": request, "images": image_list, "query": query})

@app.get("/upload/")
def upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload/")
async def upload_file(request: Request, name: str = Form(...), file: UploadFile = File(...)):
    file_location = f"public/data/{name}.jpg"
    with open(file_location, "wb+") as file_object:
        contents = await file.read()  # Here
        file_object.write(contents)

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    # save the image to the data folder before processing

    model, preprocess = clip.load("ViT-B/32",device="cpu", jit=False)
    model = model.eval()

    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor()
    ])
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    image = preprocess(image)
    image_input = torch.tensor(image)
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]
    image_input = image_input.unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        img_feature = model.encode_image(image_input).float()
        img_feature /= img_feature.norm(dim=-1, keepdim=True)

    # convert the feature vector into a numpy array
    img_feature_np = img_feature.numpy()

    # create a new pandas dataframe and append it to csv
    df = pd.DataFrame(img_feature_np, columns=[f'feature_{i}' for i in range(img_feature_np.shape[1])])
    df['file_name'] = name

    if os.path.isfile('features.csv'):
        df.to_csv('features.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('features.csv', index=False)
    return templates.TemplateResponse("upload.html",{"info": "Upload successful!", "request": request})
