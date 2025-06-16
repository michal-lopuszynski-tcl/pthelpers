SHELL=bash

visualization.md: src/visualization.ipynb
	rm -rf src/visualization_files
	quarto render src/visualization.ipynb  --to gfm
	mv src/visualization.md src/visualization_files .

src/visualization.ipynb: src/visualization.py
	jupytext --execute --to ipynb src/visualization.py

clean:
	rm -rf src/visualization.ipynb visualization_files visualization.md
