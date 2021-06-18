import streamlit as st
import cv2 as cv
import requests
import tempfile
import pandas as pd
from io import StringIO
import base64
import os

url = "http://localhost:5000/"
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href
def create_download_link(object_to_download, download_filename):
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
        b64 = base64.b64encode(object_to_download.encode()).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{download_filename}.csv">Download file</a>'
def savevideo(file):
	tmpfile = tempfile.NamedTemporaryFile(delete=False)
	tmpfile.write(file)
	capture = cv.VideoCapture(tmpfile.name)
	frame_width = int(capture.get(3))
	frame_height = int(capture.get(4))
	size = (frame_width, frame_height)
	output = cv.VideoWriter('Output.avi', cv.VideoWriter_fourcc(*'MJPG'),capture.get(cv.CAP_PROP_FPS), size)
	while(True):
		ret,frame = capture.read()
		  
		if ret == True: 
			output.write(frame)
			if cv.waitKey(1) & 0xFF == ord('s'):
				break
		else:
			break
	capture.release()
	output.release()
	cv.destroyAllWindows()

def app():
	st.title("Left Ventricle Segmenter and Ejection Fraction Predictor")
	st.markdown("Simply upload your DICOM file of a heart below, our model will then segment the left ventricle and return the ejection fraction instantly")
	datafile = st.file_uploader("Upload Video", type=['avi'])
	if st.button("Upload"):
		files = {"file": datafile.getvalue()}
		res = requests.post(url+"send_video", files=files)
		res = res.json()
		data = StringIO(res["csv"]) 
		df=pd.read_csv(data)
		df = df.to_dict()
		df = list(df.keys())[0]
		df = df.replace("\n","").replace("\r","").split(",")
		df = {str(i):[value] for i,value in enumerate(df)}
		df = pd.DataFrame(df)
		st.write("Finished uploading")
		html = create_download_link(df,"Ef")
		st.markdown(html, unsafe_allow_html=True)
		video = requests.get(url+"get_video")
		video = video.content
		savevideo(video)
		st.markdown(get_binary_file_downloader_html('Output.avi', 'Video'), unsafe_allow_html=True)
		st.write("Process Completed")

if __name__ == "__main__": 
	app()