all: init nii2stp

init:
	module load python/3-Anaconda
	source activate neuro 

nii2stp:
	python step1_nii2stp.py

testing:
	ipython -i step1_nii2stp.py --sub 4

clean:
	echo "CLEAN up temp files"

stage:
	echo "MOVE data into place"

recordenv:
	# from https://groups.google.com/a/continuum.io/forum/#!topic/conda/JKZaDDSFS6o
	echo "make sure you're in the neuro environment"
	conda list -e > env/env-conda.txt
	pip freeze > env/env-pip.txt

installenv:
	conda create -n neuro --file env/env-conda.txt
	source activate neuro
	#pip install --upgrade pip
	pip install -r env/env-pip.txt 
	echo "then install tisean"
	echo "then pip install pytisean"

generatepython:
	### from https://stackoverflow.com/questions/17077494/how-do-i-convert-a-ipython-notebook-into-a-python-file-via-commandline
	jupyter nbconvert --to script step1_nii2stp.ipynb
	### from https://stackoverflow.com/questions/10721623/echo-style-appending-but-to-the-beginning-of-a-file
	(echo "#!/usr/bin/env python"; cat step1_nii2stp.py) > step1_nii2stp.py.tmp
	mv step1_nii2stp.py.tmp step1_nii2stp.py
	chmod u+x step1_nii2stp.py
