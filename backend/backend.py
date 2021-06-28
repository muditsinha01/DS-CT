import shutil
import io
import uvicorn
import tempfile
import cv2 as cv
import os
import json
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
app = FastAPI()
@app.get("/")
async def setup():
	os.system('py setup.py')
	return "Setup Complete"
@app.get("/get_video")
async def main():
    return FileResponse('./Destination/SegmentationandEF/videos/Output.avi')

@app.post("/send_video")
def get_image(file: bytes = File(...)):
	print(type(file))
	tmpfile = tempfile.NamedTemporaryFile(delete=False)
	tmpfile.write(file)
	capture = cv.VideoCapture(tmpfile.name)
	frame_width = int(capture.get(3))
	frame_height = int(capture.get(4))
	size = (frame_width, frame_height)
	os.system("rmdir Videos /s /q")
	os.system("rmdir Destination /s /q")
	os.system("mkdir Videos")
	output = cv.VideoWriter('./Videos/Output.avi', cv.VideoWriter_fourcc(*'MJPG'),capture.get(cv.CAP_PROP_FPS), size)
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
	os.system("py run_model.py")
	with open('./Destination/cedars_ef_output.csv', 'rb') as csv:
		csv_bytes_obj = csv.read()
	csv_bytes_obj = csv_bytes_obj.decode('utf-8')
	csv_bytes_obj = json.dumps(csv_bytes_obj)
	obj = {"csv": csv_bytes_obj}
	return JSONResponse(content=obj)

if __name__ == "__main__":
	uvicorn.run("backend:app",host="0.0.0.0", port=5000, log_level="info")