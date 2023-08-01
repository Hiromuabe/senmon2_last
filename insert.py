@app.post("/insert/")
async def insert_endpoint(file: UploadFile = File(...)):
    # The same content as your existing /insert/ endpoint...

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # save the image to the data folder before processing
    image.save(os.path.join('../my-app/public/data', file.filename))

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
    df['file_name'] = file.filename

    if os.path.isfile('features.csv'):
        df.to_csv('features.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('features.csv', index=False)
    return {"file_size": len(contents)}