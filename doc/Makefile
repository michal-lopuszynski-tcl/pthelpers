# ALL

all: tracers.md visualization.md visualization_advanced.md


# TRACERS

tracers.md: src/tracers.ipynb
	rm -rf src/tracers_files tracers_files
	quarto render src/tracers.ipynb  --to gfm
	mv src/tracers.md src/tracers_files .

src/tracers.ipynb: src/tracers.py
	jupytext --execute --to ipynb src/tracers.py


# VISUALIZATION

visualization.md: src/visualization.ipynb
	rm -rf src/visualization_files visualization_files
	quarto render src/visualization.ipynb  --to gfm
	mv src/visualization.md src/visualization_files .

src/visualization.ipynb: src/visualization.py
	jupytext --execute --to ipynb src/visualization.py


# VISUALIZATION_ADVANCED

visualization_advanced.md: src/visualization_advanced.ipynb
	rm -rf src/visualization_advanced_files visualization_advanced_files
	quarto render src/visualization_advanced.ipynb  --to gfm
	mv src/visualization_advanced.md src/visualization_advanced_files .

src/visualization_advanced.ipynb: src/visualization_advanced.py
	jupytext --execute --to ipynb src/visualization_advanced.py


# CLEAN

clean:
	rm -rf src/tracers.ipynb tracers_files tracers.md
	rm -rf src/visualization.ipynb visualization_files visualization.md
	rm -rf src/visualization_advanced.ipynb visualization_advanced_files visualization_advanced.md


# MAKEFILE

Makefile: make_make.py
	python3 make_make.py > Makefile

