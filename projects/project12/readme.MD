# Project 1

### *Remote (hpc): From dtu folder -> personal folder*
Otherwise look at scp commands, the script `.load_hpc_local.sh`, should give a hint as to how that could be done

### *Local: Download data to local machine*
Use `make data_local` to download data from hpc to local machine. The download uses `scp` and you need to provide password on use of command

### *Make src package imports work*
Encountered quite a problem trying to do `import src.module`.
Best solution I found was to create a package from it and install it in my env.
The package specifics is placed in the `setup.py` file, and to install run

- `pip install -e .`
- `conda develop .`

Both are listed, but its proberbly best only to use venv or conda, so choose one ;)
