all : init nii2stp

init :
	module load python/3-Anaconda
	source activate neuro 

nii2stp :
	python step1_nii2stp.py

clean :
	echo "CLEAN up temp files"

stage :
	echo "MOVE data into place"

recordenv :
	# from https ://groups.google.com/a/continuum.io/forum/#!topic/conda/JKZaDDSFS6o
	echo "make sure you're in the neuro environment"
	conda list -e > env/env-conda.txt
	pip freeze > env/env-pip.txt

installenv :
	conda create -n neuro --file env/env-conda.txt
	source activate neuro
	pip install -r env/env-pip.txt 
	echo "Now clone and build tisean (from git@github.com:enfascination/Tisean_3.0.1frey.git)"
