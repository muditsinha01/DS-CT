import streamlit as st
import cv2 as cv
import tempfile
def app():
	st.title("Left Ventricle Segmenter and Ejection Fraction Predictor")
	st.markdown("Simply upload your DICOM file of a heart below, our model will then segment the left ventricle and return the ejection fraction instantly")
	datafile = st.file_uploader("Upload Video", type=['avi'])
	if st.button("Upload"):
		st.write(type(datafile))
		st.write("default classification 0")
if __name__ == "__main__": 
	app()