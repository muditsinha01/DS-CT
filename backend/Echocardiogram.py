import re
from pathlib import Path
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
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import math
import torch
import torchvision
import wget
import pathlib
import tempfile
import tqdm
import scipy
import shutil
sys.path.append('./dynamic/')
import echonet
#from google.colab import files
import yaml

# Path to load config file
CONFIG_PATH = "C:/Users/PC SINHA/Desktop/Final API/DS-CT/backend"
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config("configure.yaml")

#Paths for storing the processed video, pre-trained model weights and results
destinationFolder = config["destination_folder"]
videosFolder = config["video_folder"]
DestinationForWeights = config["weights_folder"]

class Preprocessing:

  def __init__(self,path):
    self.path = path

  def semiprocess(self):

       #Properties for semi-processing the video if the video is unprocessed in order to mask the data
       crpwidth=config["crp_width"]
       crpheight=config["crp_height"]
       strtwidth=config["strt_width"]
       endwidth=config["end_width"]
       strtheight=config["strt_height"]
       endheight=config["end_height"]
       
       with open("./Videos/Output.avi", "rb") as video:
        video = video.read()
       tmpfile = tempfile.NamedTemporaryFile(delete=False)
       tmpfile.write(video)
       capture = cv.VideoCapture(tmpfile.name)
       fframenum = int(capture.get(7))
       size = (int(crpwidth), int(crpheight))
       d=os.path.join(self.path, 'Output'+'.avi')
       output = cv.VideoWriter(d, cv.VideoWriter_fourcc(*'MJPG'),capture.get(cv.CAP_PROP_FPS), size)
       
       for i in range(fframenum):
        success, frame_src = capture.read()
        frame_target = frame_src[strtheight:endheight,strtwidth:endwidth]
        output.write(frame_target)
       capture.release()
       output.release()
       cv.destroyAllWindows()
  def completeprocess(self):
       with open("./Videos/Output.avi", "rb") as video:
        video = video.read()
       tmpfile = tempfile.NamedTemporaryFile(delete=False)
       tmpfile.write(video)
       capture = cv.VideoCapture(tmpfile.name)
       fframenum = int(capture.get(7))
       c=os.listdir(self.path)
       d=os.path.join(self.path,'Output'+'.avi')
       output = cv.VideoWriter(d, cv.VideoWriter_fourcc(*'MJPG'),capture.get(cv.CAP_PROP_FPS), (112,112))
       for i in range(fframenum):
            ret,frame=capture.read()
            if ret==True:
                 framesw = cv.resize(frame,(112,112), interpolation = cv.INTER_CUBIC)
                 output.write(framesw)
            else:
                 break
       print("The video has been processed")

  def preprocess(self):
    ifpa=os.listdir(self.path)
    ifn=os.path.join(self.path, ifpa[0])
    checkvideo = cv.VideoCapture(ifn)
    width  = int(checkvideo.get(3))
    height = int(checkvideo.get(4))
    if(width==112 and height==112):
      print("The video doesn't require pre-processing")
    else:
      print("The video requires pre-processing")
      self.semiprocess()
      self.completeprocess()

class EchoCardiogram:

  #downloading the pre-trained models with weights
  segmentationWeightsURL = config["seg_weights_url"]
  ejectionFractionWeightsURL = config["ejection_url"]
  
  def __init__(self, destination, inputdir, modeldir):
    self.destination = destination
    self.inputdir = inputdir
    self.modeldir = modeldir
    self.createdir()
    self.downloadmodel()
  
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
  
  #Properties for Echocardiogram EjectionFraction
  frames = config["frames"]
  period = config["period"]
  batch_size = config["batch_size"]
  
  def __init__(self, destination, inputdir, modeldir):
    super().__init__(destination, inputdir, modeldir)

  def checkEnivornment(self):
    self.model = torchvision.models.video.r2plus1d_18(pretrained=False)
    self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
    self.device = torch.device("cpu")
    self.checkpoint = torch.load(os.path.join(self.modeldir, os.path.basename(self.ejectionFractionWeightsURL)), map_location = "cpu")
    self.state_dict_cpu = {k[7:]: v for (k, v) in self.checkpoint['state_dict'].items()}
    self.model.load_state_dict(self.state_dict_cpu)

  def dataLoader(self):
    ds = echonet.datasets.Echo(split = "external_test", external_test_location = self.inputdir)
    mean, std = echonet.utils.get_mean_and_std(ds)
    kwargs = {"target_type": "EF", "mean": mean, "std": std, "length": self.frames, "period": self.period}
    self.ds = echonet.datasets.Echo(split = "external_test", external_test_location = self.inputdir, **kwargs)
    self.test_dataloader = torch.utils.data.DataLoader(self.ds, batch_size = 1, num_workers = 0, shuffle = True, pin_memory=(self.device.type == 'cpu')) 

  def predict(self):
    self.loss, self.yhat, self.y = echonet.utils.video.run_epoch(self.model, self.test_dataloader, False, None, device=self.device, save_all=True)
  
  def outputFile(self):
    self.destination = "./Destination/EjectionFractionResults"
    if not os.path.exists(self.destination):
      os.mkdir(self.destination)
    self.path = './Videos/'
    filnm=os.listdir(self.path)
    output = os.path.join(self.destination, "EjectionFraction"+'.csv')
    with open(output, "w") as g:
        for (filename, pred) in zip(self.ds.fnames, self.yhat):
            for (i,p) in enumerate(pred):
                g.write("{},{:.4f}\n".format(filename,p))

  def rm(self):
    shutil.rmtree(self.inputdir)
    print("\n Un-proccesed video is in "+str(Path(inputfilepath))+" folder")
    directory= os.listdir(semiprocessedfilepath)
    if len(directory)==0:
      print("\n " +str(Path(semiprocessedfilepath))+" folder will be empty since there was no pre-processing of video")
      print("\n Completely-processed video will have the same video as in "+str(Path(inputfilepath))+" folder since there was no pre-processing of video")
    else:
      print("\n Semi-processed video is in "+str(Path(semiprocessedfilepath))+" folder")
      print("\n Completely-processed video is in "+str(Path(completelyprocessedfilepath))+" folder")
    print("\n The Ejection Fraction Results and Segmentation Results are stored in "+str(Path(destinationFolder))+" folder")
    print("\n Completed!")  

class Segmentation(EchoCardiogram):

  def __init__(self, destination, inputdir, modeldir):
    super().__init__(destination, inputdir, modeldir)
    self.destinationFolder()
    
  def destinationFolder(self):
    self.destination = config["destination_sg"]
    if not os.path.exists(self.destination):
      os.mkdir(self.destination)
    else:
      print("Already Present")

  def checkenvironment1(self):
    self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=False)
    self.model.classifier[-1] = torch.nn.Conv2d(self.model.classifier[-1].in_channels, 1, kernel_size=self.model.classifier[-1].kernel_size)
    self.device = torch.device("cpu")
    self.checkpoint = torch.load(os.path.join(self.modeldir, os.path.basename(self.segmentationWeightsURL)))
    self.state_dict_cpu = {k[7:]: v for (k, v) in self.checkpoint['state_dict'].items()}
    self.model.load_state_dict(self.state_dict_cpu)
    torch.cuda.empty_cache()
  
  def collate_fn(self, x):
    x, f = zip(*x)
    i = list(map(lambda t: t.shape[1], x))
    x = torch.as_tensor(np.swapaxes(np.concatenate(x, 1), 0, 1))
    return x, f, i
  
  def smf(self):
    ds = echonet.datasets.Echo(split = "external_test", external_test_location = self.inputdir)
    mean, std = echonet.utils.get_mean_and_std(ds)
    self.ds = echonet.datasets.Echo(split = "external_test", external_test_location = self.inputdir, target_type=["Filename"], length=None, period=1, mean = mean, std = std)
    dataloader = torch.utils.data.DataLoader(self.ds, batch_size = 1, num_workers = 0, shuffle = False, pin_memory=(self.device.type == 'cpu'),collate_fn=self.collate_fn)
    
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
    rd_video=cv.VideoCapture('./Videos/Output.avi')
    fps = int(rd_video.get(cv.CAP_PROP_FPS))
    dataloader = torch.utils.data.DataLoader(echonet.datasets.Echo(split="external_test", external_test_location = self.inputdir, target_type=["Filename"], length=None, period=1),
                                            batch_size=1, num_workers=0, shuffle=False, pin_memory=False)
    
    if not all(os.path.isfile(os.path.join(self.destination, "videos", f)) for f in dataloader.dataset.fnames):
        pathlib.Path(os.path.join(self.destination, "videos")).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(self.destination, "size")).mkdir(parents=True, exist_ok=True)
        echonet.utils.latexify()
        with open(os.path.join(self.destination, "size.csv"), "w") as g:
            g.write("Filename,Frame,Size,ComputerSmall\n")
            for (x, filename) in tqdm.tqdm(dataloader):
                x = x.numpy()
                for i in range(len(filename)):
                    file = pathlib.Path(filename[i]).stem
                    y = np.load(os.path.join(self.destination, "labels", file + ".npy"))
                    img = x[i, :, :, :, :].copy()
                    logit=y.copy()
                    img[1, :, :, :] = img[0, :, :, :]
                    img[2, :, :, :] = img[0, :, :, :]
                    img = np.concatenate((img, img), 3)
                    img[0, :, :, 112:] = np.maximum(255. * (logit > 0), img[0, :, :, 112:])

                    img = np.concatenate((img, np.zeros_like(img)), 2)
                    size = (logit > 0).sum((1, 2))
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
                    self.path = './Videos/'
                    filnm=os.listdir(self.path)

                    echonet.utils.savevideo(os.path.join(self.destination, "videos", filnm[0]), img.astype(np.uint8), fps)
                    print("\n")

class EfSe(Segmentation,Ejectionfraction):

  def __init__(self, destination, inputdir, modeldir):
    super().__init__(destination, inputdir, modeldir)

  def main(self):
    self.checkEnivornment()
    self.dataLoader()
    self.predict()
    self.outputFile()
    self.destinationFolder()
    self.checkenvironment1()
    self.smf()
    self.outputvideo()