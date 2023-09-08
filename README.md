
## TODO
- given list of values for metabolite, calculate derivatives so that metabolite mass follows the values 

## NOTE
[The pyrightconfig.json](https://stackoverflow.com/questions/74510279/pyright-cant-see-poetry-dependencies
) file is used for my specific IDE, not needed in general.


## NOTE
Using ```VisiData``` is a very useful way to view xlsx and csv files when writing code:
```
$ vd name.xlsx
```

## Profiling Code
To profile the code, i.e. to find how long each part of the code takes to run so as to optimize runtime, you can run 
```
poetry run python -m cProfile metabolicpd/simulation.py
```
to print the data to stdout. In order to print this information to a file (which is usually the best idea), use this
```
poetry run python -m cProfile -o simulation_stats metabolicpd/simulation.py
```
With a profile file, we can use the pstats module, e.g. using the repl, to analyse the data.
See [this](https://docs.python.org/3/library/profile.html) link for documentation and examples

# KEGG Notes
- [KGML Docs](https://www.kegg.jp/kegg/xml/docs/)
- [Pathway with gene information](https://www.genome.jp/kegg-bin/show_pathway?mtu01200))
- [Pathway without gene info](https://www.genome.jp/kegg-bin/show_pathway?rn01200))
