
## TODO
- given list of values for metabolite, calculate derivatives so that metabolite mass follows the values 


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
