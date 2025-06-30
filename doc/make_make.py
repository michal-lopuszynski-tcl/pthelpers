#!/usr/bin/env pytho3

name_roots = ["tracers", "visualization", "visualization_advanced"]


def make_target_all(name_roots):
    return "all: " + " ".join([ f"{s}.md" for s in name_roots]) + "\n"


def make_target_clean(name_roots):
    s = "clean:\n"
    for n in name_roots:
        s += f"\trm -rf src/{n}.ipynb {n}_files {n}.md\n"
    return s


def make_target_md(n):
    s = \
f"""
{n}.md: src/{n}.ipynb
\trm -rf src/{n}_files {n}_files
\tquarto render src/{n}.ipynb  --to gfm
\tmv src/{n}.md src/{n}_files .

src/{n}.ipynb: src/{n}.py
\tjupytext --execute --to ipynb src/{n}.py
"""
    return s


def make_target_makefile():
    return "Makefile: make_make.py\n\tpython3 make_make.py > Makefile\n"


if __name__ == "__main__" :
    s = "# ALL\n\n"
    s += make_target_all(name_roots)
    s += "\n\n"

    for n in name_roots:
        s += f"# {n.upper()}\n"
        s += make_target_md(n)
        s += "\n\n"

    s += "# CLEAN\n\n"
    s += make_target_clean(name_roots)
    s += "\n\n"

    s += "# MAKEFILE\n\n"
    s += make_target_makefile()

    print(s)
