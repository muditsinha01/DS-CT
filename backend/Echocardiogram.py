import re
import os, os.path
from os.path import splitext
import pydicom as dicom
import numpy as np
from pydicom.uid import UID, generate_uid
from multiprocessing import dummy as multiprocessing
import time
import subprocess
import datetime
from datetime import date
import sys
import cv2
import matplotlib.pyplot as plt
import sys
import math
import torch
import torchvision
import wget
import pathlib
import tqdm
import scipy
import shutil
sys.path.append("./dynamic/")
import echonet
#import files

destinationFolder = "./Destination/"
videosFolder = "./Videos/"
DestinationForWeights = "./Pre_Trained_Modelswithweights/"

class EchoCardiogram:

  segmentationWeightsURL = 'https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/deeplabv3_resnet50_random.pt'
  ejectionFractionWeightsURL = 'https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt'
  
  def __init__(self, destination, inputdir, modeldir):
    self.destination = destination
    self.inputdir = inputdir
    self.modeldir = modeldir
    self.createdir()
    self.downloadmodel()
    #self.uploadvideo()
  
  def createdir(self):
    if not os.path.exists(self.destination):
      os.mkdir(self.destination)
    else:
      print("Already Present")

    if not os.path.exists(self.inputdir):
      os.mkdir(self.inputdir)
    else:
      print("Already Present")

    if not os.path.exists(self.modeldir):
      os.mkdir(self.modeldir)
    else:
      print("Already Present")

  def uploadvideo(self):
    os.chdir(self.inputdir)
    uploaded = files.upload()
    os.chdir('..')

  def downloadmodel(self):
    if not os.path.exists(os.path.join(self.modeldir, os.path.basename(self.segmentationWeightsURL))):
      print("Downloading Segmentation Weights, ", self.segmentationWeightsURL," to ",os.path.join(self.modeldir,os.path.basename(self.segmentationWeightsURL)))
      filename = wget.download(self.segmentationWeightsURL, out = self.modeldir)
    else:
        print("Segmentation Weights already present")
        
    if not os.path.exists(os.path.join(self.modeldir, os.path.basename(self.ejectionFractionWeightsURL))):
        print("Downloading EF Weights, ", self.ejectionFractionWeightsURL," to ",os.path.join(self.modeldir,os.path.basename(self.ejectionFractionWeightsURL)))
        filename = wget.download(self.ejectionFractionWeightsURL, out = self.modeldir)
    else:
        print("EF Weights already present")

class Ejectionfraction(EchoCardiogram):
  
  frames = 32
  period = 1
  batch_size = 20
  
  def __init__(self, destination, inputdir, modeldir):
    super().__init__(destination, inputdir, modeldir)

  def checkEnivornment(self):
    self.model = torchvision.models.video.r2plus1d_18(pretrained=False)
    self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
    if torch.cuda.is_available():
        print("cuda is available, original weights")
        self.device = torch.device("cuda")
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.checkpoint = torch.load(os.path.join(self.modeldir, os.path.basename(self.ejectionFractionWeightsURL)))
        self.model.load_state_dict(self.checkpoint['state_dict'])
    
    else:
        print("cuda is not available, cpu weights")
        self.device = torch.device("cpu")
        self.checkpoint = torch.load(os.path.join(self.modeldir, os.path.basename(self.ejectionFractionWeightsURL)), map_location = "cpu")
        self.state_dict_cpu = {k[7:]: v for (k, v) in self.checkpoint['state_dict'].items()}
        self.model.load_state_dict(self.state_dict_cpu)

  def dataLoader(self):
    ds = echonet.datasets.Echo(split = "external_test", external_test_location = self.inputdir)
    mean, std = echonet.utils.get_mean_and_std(ds)
    kwargs = {"target_type": "EF", "mean": mean, "std": std, "length": self.frames, "period": self.period}
    self.ds = echonet.datasets.Echo(split = "external_test", external_test_location = self.inputdir, **kwargs)
    self.test_dataloader = torch.utils.data.DataLoader(self.ds, batch_size = 1, num_workers = 5, shuffle = True, pin_memory=(self.device.type == "cuda")) 

  def predict(self):
    self.loss, self.yhat, self.y = echonet.utils.video.run_epoch(self.model, self.test_dataloader, False, None, device=self.device, save_all=True)
  
  def outputFile(self):
    self.destination = "./Destination/"
    output = os.path.join(self.destination, "cedars_ef_output.csv")
    with open(output, "w") as g:
        for (filename, pred) in zip(self.ds.fnames, self.yhat):
            for (i,p) in enumerate(pred):
                g.write("{},{},{:.4f}\n".format(filename, i, p))
    shutil.rmtree(self.inputdir)
    print("\n Completed!")

class Segmentation(EchoCardiogram):

  def __init__(self, destination, inputdir, modeldir):
    super().__init__(destination, inputdir, modeldir)
    self.destinationFolder()
    
  def destinationFolder(self):
    self.destination = "./Destination/SegmentationandEF/"
    if not os.path.exists(self.destination):
      os.mkdir(self.destination)
    else:
      print("Already Present")

  def checkenvironment1(self):
    self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=False)
    self.model.classifier[-1] = torch.nn.Conv2d(self.model.classifier[-1].in_channels, 1, kernel_size=self.model.classifier[-1].kernel_size)
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
      self.model = torch.nn.DataParallel(self.model)
      self.model.to(self.device)
      self.checkpoint = torch.load(os.path.join(self.modeldir, os.path.basename(self.segmentationWeightsURL)))
      self.model.load_state_dict(self.checkpoint['state_dict'])
      torch.cuda.empty_cache()
  
  def collate_fn(self, x):
    x, f = zip(*x)
    i = list(map(lambda t: t.shape[1], x))
    x = torch.as_tensor(np.swapaxes(np.concatenate(x, 1), 0, 1))
    return x, f, i
  
  def smf(self):
    ds = echonet.datasets.Echo(split = "external_test", external_test_location = self.inputdir)
    mean, std = echonet.utils.get_mean_and_std(ds)
    dataloader = torch.utils.data.DataLoader(echonet.datasets.Echo(split="external_test", external_test_location = self.inputdir, target_type=["Filename"], length=None, period=1, mean=mean, std=std),
                                          batch_size=10, num_workers=0, shuffle=False, pin_memory=(self.device.type == "cuda"), collate_fn = self.collate_fn) 
    
    if not all([os.path.isfile(os.path.join(self.destination, "labels", os.path.splitext(f)[0] + ".npy")) for f in dataloader.dataset.fnames]):
      pathlib.Path(os.path.join(self.destination, "labels")).mkdir(parents=True, exist_ok=True)
      block = 1024
      self.model.eval()

      with torch.no_grad():
          for (x, f, i) in tqdm.tqdm(dataloader):
              x = x.to(self.device)
              y = np.concatenate([self.model(x[i:(i + block), :, :, :])["out"].detach().cpu().numpy() for i in range(0, x.shape[0], block)]).astype(np.float16)
              start = 0
              for (filename, offset) in zip(f, i):
                  np.save(os.path.join(self.destination, "labels", os.path.splitext(filename)[0]), y[start:(start + offset), 0, :, :])
                  start += offset
  

  def outputvideo(self):
    dataloader = torch.utils.data.DataLoader(echonet.datasets.Echo(split="external_test", external_test_location = self.inputdir, target_type=["Filename"], length=None, period=1),
                                            batch_size=1, num_workers=8, shuffle=False, pin_memory=False)
    
    if not all(os.path.isfile(os.path.join(self.destination, "videos", f)) for f in dataloader.dataset.fnames):
        pathlib.Path(os.path.join(self.destination, "videos")).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(self.destination, "size")).mkdir(parents=True, exist_ok=True)
        echonet.utils.latexify()
        with open(os.path.join(self.destination, "size.csv"), "w") as g:
            g.write("Filename,Frame,Size,ComputerSmall\n")
            for (x, filename) in tqdm.tqdm(dataloader):
                x = x.numpy()
                for i in range(len(filename)):
                    img = x[i, :, :, :, :].copy()
                    logit = img[2, :, :, :].copy()
                    img[1, :, :, :] = img[0, :, :, :]
                    img[2, :, :, :] = img[0, :, :, :]
                    img = np.concatenate((img, img), 3)
                    img[0, :, :, 112:] = np.maximum(255. * (logit > 0), img[0, :, :, 112:])

                    img = np.concatenate((img, np.zeros_like(img)), 2)
                    size = (logit > 0).sum(2).sum(1)
                    try:
                        trim_min = sorted(size)[round(len(size) ** 0.05)]
                    except:
                        import code; code.interact(local=dict(globals(), **locals()))
                    trim_max = sorted(size)[round(len(size) ** 0.95)]
                    trim_range = trim_max - trim_min
                    peaks = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])
                    for (x, y) in enumerate(size):
                        g.write("{},{},{},{}\n".format(filename[0], x, y, 1 if x in peaks else 0))
                    fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                    plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                    ylim = plt.ylim()
                    for p in peaks:
                        plt.plot(np.array([p, p]) / 50, ylim, linewidth=1)
                    plt.ylim(ylim)
                    plt.title(os.path.splitext(filename[i])[0])
                    plt.xlabel("Seconds")
                    plt.ylabel("Size (pixels)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.destination, "size", os.path.splitext(filename[i])[0] + ".pdf"))
                    plt.close(fig)
                    size -= size.min()
                    size = size / size.max()
                    size = 1 - size
                    for (x, y) in enumerate(size):
                        img[:, :, int(round(115 + 100 * y)), int(round(x / len(size) * 200 + 10))] = 255.
                        interval = np.array([-3, -2, -1, 0, 1, 2, 3])
                        for a in interval:
                            for b in interval:
                                img[:, x, a + int(round(115 + 100 * y)), b + int(round(x / len(size) * 200 + 10))] = 255.
                        if x in peaks:
                            img[:, :, 200:225, b + int(round(x / len(size) * 200 + 10))] = 255.
                    echonet.utils.savevideo(os.path.join(self.destination, "videos", filename[i]), img.astype(np.uint8), 50)

class EfSe(Segmentation,Ejectionfraction):

  def __init__(self, destination, inputdir, modeldir):
    super().__init__(destination, inputdir, modeldir)

  def main(self):
    self.checkEnivornment()
    self.dataLoader()
    self.predict()
    self.destinationFolder()
    self.checkenvironment1()
    self.smf()
    self.outputvideo()
    self.outputFile()