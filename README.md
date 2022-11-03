# Agrippa

This python package is meant to assist in building/understanding/analyzing machine learning models. The core of the system is a markup language that can be used to specify a model architecture. This package contains utilities to convert that language into the ONNX format, which is compatibly with a variety of deployment options and ML frameworks.

# Installation

Agrippa can be installed with `pip install agrippa`. The `requirements.txt` file contains dependencies to run both the package and the tests found in the `tests` folder.

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
        suppress=False
    ):
```

The outfile parameter defaults to the infile with a .onnx extension. The `suppress` variable controls whether export will print anything.

# Markup Language Spec

A project should be bundled into its own directory, which should have three files:
1. One file with the extension `.agr` or `.xml` specifying the model architecture
2. A `weights.pkl` file to specify the parameter values in the model (optional)
3. A `meta.json` file to define certain metadata, like the producer name (optional)

The architecture file is parsed like XML, so it should be well-formed XML. Recall that tags with no respective end-tag should end with ` \>`, and all attributes should be formatted like strings with quotes around them.

## Markup Syntax

Every architecture file should be encased in a `<model>` tag, ideally with an attribute `script-version="0.0.0"` (the current version).

There are only three types of root-level tags that are allowed: `<import>`, `<export>`, and `<block>`. The import and export tags specify the inputs and outputs of the entire model, respectively. There may be multiple of each type, but each type must appear at least once. Each import and export tag must have three attributes: `dim`, `from`, and `type`. They are used like so:

```
<import dim="[3, 1]" from="input" type="float32" />
<export dim="[3, 1]" from="y" type="float32" />
```

The `from` name for the export matches the name you should expect from ONNX runtime. It should also match the output of the last node from which you are piping output.

Most of the architechture should be contained inside `<block>` tags. These tags take a title attribute, which does not need to be unique. Importantly, `<node>` tags must be inside blocks. Block tags should contain `<import>` and `<export>` tags (with the attributes mentioned above) specifying all of the inputs/outputs the underlying nodes inside the block use.

Nodes define operations. Their `op` attribute defines the ONNX op type they will be converted to. They must also have a `title` attribute, which is unique. Nodes must also contain appropriate `<input>`, `<output>`, and `<params>` tags. The `<input>` and `<params>` tags need to be in the order specified in the ONNX documentation for a particular node type. See an example node:

```
<node title="Linear" op="MatMul">
    <params dim="[3, 3]" name="W" type="float32" shared="no" />
    <input dim="[3, 1]" src="input" />
    <output dim="[3, 1]" name="linear" />
</node>
```

Parameters, which are specified using the `<params>` tag, take a `name` attribute (unique only for non-shared parameters), a `dim` attribute, a `type` attribute, and an optional `shared` attribute. The `shared` attribute should equal "yes" or "no".

## Repetitions

Blocks may take a `rep` attribute, which defines how many times a block should be stacked on top of itself. Its outputs are passed to its inputs and so on. The number of inputs and the number of outputs need not match (they are matched based on order; note that if you want to use intermediate outputs, you must account for name mangling in repeated blocks). Even though the names of the outputs are mangled during repetitions, you may use the outputs in your markup with consideration to that fact: simply refer back to the name you specified, which is automatically mapped to the last name in the repetition.

## Other Rules

Each node in your file must have a unique title (name in ONNX). If it is inside a repeated block, the title will be mangled when it is converted to ONNX. Similarly, repeated output names will also be mangled. Parameter names should be unique only when they are not shared parameters; parameters inside repeated blocks will have their names mangled. Currently, name mangling simply appends an index to the name starting with 1. Name mangling affects parameters, node titles, and output/input names separately.

Any behavior not mentioned here is undefined.

## Supported Types

The only currently supported type is `float32`.

## Supported ONNX OpTypes

The currently supported op types are:

- Add
- MatMul
- LpNormalization
- Relu

## Syntax Highlighting in VSCode

If you'd like to use the extension `.agr` for clarity, you can enable syntax highlighting in vscode by placing the following in a settings.json file:
```
"files.associations": {
    "*.agr": "xml"
}
```
To create that settings file, use the command pallet (CTRL-SHIFT-P), type `settings.json`, and choose the appropriate option.

# Examples

You can find example projects inside the `tests` folder.