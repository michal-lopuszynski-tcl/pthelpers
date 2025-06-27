SHELL=bash

all: visualization.md tracers.md


visualization.md: src/visualization.ipynb
	rm -rf src/visualization_files visualization_files
	quarto render src/visualization.ipynb  --to gfm
	mv src/visualization.md src/visualization_files .

src/visualization.ipynb: src/visualization.py
	jupytext --execute --to ipynb src/visualization.py


tracers.md: src/tracers.ipynb
	rm -rf src/tracers_files tracers_files
	quarto render src/tracers.ipynb  --to gfm
	mv src/tracers.md src/tracers_files .

src/tracers.ipynb: src/tracers.py
	jupytext --execute --to ipynb src/tracers.py


clean:
	rm -rf src/visualization.ipynb visualization_files visualization.md
	rm -rf src/tracers.ipynb tracers_files tracers.md
