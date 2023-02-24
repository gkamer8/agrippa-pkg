# Agrippa

This python package is meant to assist in building/understanding/analyzing machine learning models. The core of the system is a markup language that can be used to specify a model architecture. This package contains utilities to convert that language into the ONNX format, which is compatible with a variety of deployment options and ML frameworks.

# Installation

Agrippa can be installed with `pip install agrippa`. The `requirements.txt` file contains dependencies to run both the package and the tests found in the `tests` folder.

If you'd like to use the latest development version or contribute, you can clone this repo and run it in a virtual environment using:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

on 'nix based systems.

# Usage

The principal function is export, which takes a project folder and compiles the contents into a .onnx file.

```
import agrippa

model_dir = '../path/to/dir'
agrippa.export(model_dir, 'outfile_name.onnx')
```

The function header for export is:

```
def export(
        infile,
        outfile=None,
        producer="Unknown",
        graph_name="Unknown",
        write_weights=True,
        suppress=False,
        reinit=False,
        bindings=None,
        log=False,  # Write a log file for the compilation
        log_filename=LOG_FILENAME,
        index=None  # Main file that things are imported into
    ):
```

# Docs

Documentation is available on the [Agrippa website](https://agrippa.build/docs) under "Docs".

# Examples

Examples of usage are available in the `tests` folder and on the [Agrippa website](https://agrippa.build).