"""Temel ve ileri görüntü işleme işlemlerini sentetik görüntüde uygular."""
from pathlib import Path
import json
import numpy as np
from scipy import ndimage

ROOT=Path(__file__).resolve().parent; RESULTS=ROOT/"results"
def synthetic_image(size=96):
    y,x=np.mgrid[:size,:size]; image=np.zeros((size,size,3),dtype=float); face=(x-size/2)**2+(y-size/2)**2<(size*.32)**2; image[face]=[.85,.65,.45]; image[30:38,30:40]=.05; image[30:38,56:66]=.05; image[62:67,38:58]=[.4,.05,.05]; return image
def grayscale(image): return image@np.array([.299,.587,.114])
def edges(gray):
    sx=ndimage.sobel(gray,axis=1); sy=ndimage.sobel(gray,axis=0); return np.hypot(sx,sy)
def connected_regions(mask,min_pixels=20):
    labels,count=ndimage.label(mask); objects=ndimage.find_objects(labels); boxes=[]
    for label,slices in enumerate(objects,start=1):
        if slices and np.sum(labels[slices]==label)>=min_pixels: boxes.append([[s.start,s.stop] for s in slices])
    return boxes
def classification_features(image):
    gray=grayscale(image); edge=edges(gray); return {"brightness":float(gray.mean()),"contrast":float(gray.std()),"edge_density":float((edge>.35).mean()),"red_ratio":float(image[:,:,0].mean()/(image.mean()+1e-9))}
def main():
    image=synthetic_image(); gray=grayscale(image); edge=edges(gray); result={"image_shape":list(image.shape),"regions":connected_regions(edge>.35),"features":classification_features(image)}; RESULTS.mkdir(exist_ok=True); (RESULTS/"image_analysis.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print("GÖRÜNTÜ İŞLEME\n"+"="*25); print(result)
if __name__=="__main__": main()
