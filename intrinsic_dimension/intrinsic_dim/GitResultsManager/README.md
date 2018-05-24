GitResultsManager
=====================

Note: GitResultsManager does not profess to be remotely operable on
any operating system besides Linux and Mac. Evidence of success on
other OSs is appreciated.



Installing
---------------------

### One line global install:

    git clone https://github.com/yosinski/GitResultsManager.git && \
    cd GitResultsManager && \
    sudo python setup.py install && \
    sudo cp resman resman-td git-recreate /usr/local/bin/

Replace `/usr/local/bin` with another location on your path, if desired. If installing the Python packages in your home directory (perhaps using virtualenv), you should omit the first `sudo`, and if installing scripts in your home directory, skip the second.



Usage
---------------------

GitResultsManager may be used in two ways:

1. **(recommended)** Using the `resman` wrapper script to run programs in any language, or
2. From within Python as a Python module.

(1) is more general, while (2) offers more control. The following examples are available in the `examples` directory.

### Example of using `resman` wrapper script to run a C program:

First, we'll compile the `demo-c` program (from the examples directory) and run it without `resman`:

    g++ -o demo-c demo-c.cc   # compile program first if necessary
    ./demo-c

Output:

    Environment variable GIT_RESULTS_MANAGER_DIR is undefined. To demonstrate logging, run this instead as
        resman junk ./demo-c
    This line is logged
    This line is logged (stderr)
    This line is logged
    This line is logged (stderr)
    This line is logged
    This line is logged (stderr)

Notice that it complains it cannot find the GIT_RESULTS_MANAGER_DIR
environment variable. This is how the program knows it is not being
run from within `resman`. Now, try using `resman` to run it:

    resman run-name ./demo-c

Output:

    WARNING: GitResultsManager running in GIT_DISABLED mode: no git information saved! (Is /Users/jason/temp/examples in a git repo?)
      Logging directory: results/121030_183101_run-name
            Command run: ./demo-c
               Hostname: lapaz
      Working directory: /Users/jason/temp/examples
    The current GIT_RESULTS_MANAGER_DIR is: results/121030_183101_run-name
    This line is logged
    This line is logged
    This line is logged
    This line is logged (stderr)
    This line is logged (stderr)
    This line is logged (stderr)
           Wall time:  0.024
      Processor time:  0.012

Notice how `resman` adds a few lines of information to the beginning and ending of the output? Looking at each line in order:

    WARNING: GitResultsManager running in GIT_DISABLED mode: no git information saved! (Is /Users/jason/temp/examples in a git repo?)

Warning because we aren't running from within a git repository, removing most of the usefulness of GitResultsManager.

      Logging directory: results/121030_183101_run-name

The directory that was created for this run, in the format `<datestamp>_<timestamp>_<name of run>`

            Command run: ./demo-c

Which command you actually ran.

               Hostname: lapaz

The host this run was performed on (useful when running on clusters or
multiple machines with non-identical configurations)

      Working directory: /Users/jason/temp/examples

The working directory. Next follows the actual output of the program, and then at the end...

           Wall time:  0.024
      Processor time:  0.012

`resman` notes how long the program took to execute in wall time and processor time.



### Simple code change to use `resman` wrapper script in Python:

Import the `os` module:

    import os

Check if we're running from within `resman`. If so, use the directory `resman` provides, else save output to the current directory:

    try:
        savedir = os.environ['GIT_RESULTS_MANAGER_DIR']
    except KeyError:
        savedir = '.'

    # later in code, when saving plots / etc:

    savefig(os.path.join(savedir, 'myplot.png'))



### Example of using the `GitResultsManager` class within Python.

See `examples/demo-GRM-module.py`.



Development task list
----------------------

### To do

1. Add settings override via `~/.config/gitresultsmanager_config.py` or similar
1. Documentation

Want to help? Pull requests are welcome!
