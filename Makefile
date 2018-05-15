run:
	source activate neuro 
	python step1_nii2stp.py

clean:
	echo "CLEAN up temp files"

stage:
	echo "MOVE data into place"
