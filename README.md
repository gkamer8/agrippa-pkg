# Agrippa

This python package is meant to assist in building/understanding/analyzing machine learning models. The core of the system is a markup language that can be used to specify a model architecture. This package contains utilities to convert that language into the ONNX format, which is compatible with a variety of deployment options and ML frameworks.

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
        infile,  # the markup file
        outfile=None,  # the .onnx file name
        producer="Unknown",  # your name
        graph_name="Unknown",  # the name of the model
        write_weights=True,  # should this create a weights file, should none exist
        suppress=False,  # suppresses certain print statements
        reinit=False,  # reinitializes all weights from the weights file
        bindings=None  # a dictionary to bind variables present in the markup
    ):
```

# Markup Language Spec

A project should be bundled into its own directory, which should have three files:
1. One file with the extension `.agr` or `.xml` specifying the model architecture
2. A `weights.pkl` file to specify the parameter values in the model (optional)
3. A `meta.json` file to define certain metadata, like the producer name (optional)

The architecture file is parsed like XML, so it should be well-formed XML. Recall that tags with no respective end-tag should end with ` \>`, and all attributes should be formatted like strings with quotes around them.

## Markup Syntax

Every architecture file should be encased in a `<model>` tag, ideally with an attribute `script-version="0.0.0"` (the current version).

An example script that does only one matrix multiply might look like this:

```
<model script-version="0.0.1">
    <import dim="[5, 1]" from="input" type="float32" />
    <block title="Projection">
        <import from="input" />
        <node title="Linear" op="MatMul">
            <params dim="[5, 5]" name="W" type="float32" />
            <input dim="[var(features), 1]" src="input" />
            <output dim="[5, 1]" name="y" />
        </node>
        <export from="y" />
    </block>
    <export dim="[5, 1]" from="y" type="float32" />
</model>
```

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

Parameters, which are specified using the `<params>` tag, take a `name` attribute (unique only for non-shared parameters), a `dim` attribute, a `type` attribute, and an optional `shared` attribute. The `shared` attribute should equal "yes" or "no". It specifies whether a parameter name is meant to be unique; by default, parameters which share the same name (such as in a repitition) become independent values upon compilation.

## Repetitions

Blocks may take a `rep` attribute, which defines how many times a block should be stacked on top of itself. Its outputs are passed to its inputs and so on. The number of inputs and the number of outputs need not match (they are matched based on order; note that if you want to use intermediate outputs, you must account for name mangling in repeated blocks). Even though the names of the outputs are mangled during repetitions, you may use the outputs in your markup with consideration to that fact: simply refer back to the name you specified, which is automatically mapped to the last name in the repetition.

## Variable Bindings

The `agrippa.export` function takes an optional argument, `bindings`. The `bindings` parameter is meant to be a dictionary of variables, set by the user, to replace areas in the markup file where the `var` function is used. For example, if an input tag has a `dim` attribute set to `"[var(image_width), var(image_height)]"`, a binding of `{'image_width': 512, 'image_height': '256'}` would set all occurances of `var(image_height)` to `512` and all occurances of `var(image_height)` to `256`. Note that in all cases, strings are used, since xml attributes require strings; the values are type-casted upon compilation.

## Expressions

Attributes also support expressions using `expr()`. For example, in order to specify that a parameter should be initialized to the square of a variable (supplied in bindings), you could use:
`<params name="squared" dim="[2, 2]" init="constant" init_args="[expr(my_var^2)]">`.

Note that the expression goes inside the list (expressions do not support lists). They support to following binary operators: `^`, `*`, `/`, `%`, `-`, `+`.

Also note that `expr(my_var)` and `var(my_var)` are equivalent.

## Weight Initialization

By default, weights are initialized with a standard normal distribution. There are ways to specify other initializations for each parameter, however. The `params` tag takes an optional `init` attribute along with an optional `init_args` attribute. The `init_args` attribute must always be some value (non-string), such as a list (e.g., `init_args="[2, 3]"`). Recall that all attributes are specified with double quotation marks) The options for initialization are:

| Value    | Description    | Arguments |
|----------|----------------|-----------|
| normal   | Normally distributed | A list of two numbers, the first defining the mean and the second defining the standard deviation. |
| uni_random  | Uniformly random in [a, b) | A list of two numbers, the first defining the a and the second defining b.|
| zeros | All zeros | None |
| ones  | All ones  | None |
| constant | Initializes tensor to specified value | The first argument in the list is the value

## Frozen Parameters

In order to freeze a parameter, you can set the `frozen` attribute equal to `yes`. Internally, this option adds `$constant` to the ONNX initialization names. When importing the parameter into PyTorch using the conversion tool, the `$constant` indicates that the initializer should be added as a buffer (constant) rather than a parameter.

## Importing From Other Files

Another file can be used in your model by using a `block` tag with a `src` attribute. Like so:

```<block src="path/to/file.agr" name="imported_file" />```

The `name` attribute defines how you refer to imports/exports of the imported model. For example, if the linked model has a root level import with name `inputs`, an output (or import) (in the original file) with name `imported_file$inputs` will be automatically passed to the imported model. Likewise, an export can be referred to in the original file by specifying an input with name `imported_file$out_name_from_imp_file`.

## Other Rules

### Names

Node titles are optional (a default, unique game is given to them upon compilation). Parameter names should be unique only when they are not shared parameters; parameters inside repeated blocks will have their names mangled so that they are unique. Name mangling affects parameters, node titles, and output/input names separately.

### Types

Types by default are set to float32.

### Dimensions

Specifying the dimensions of inputs and outputs are optional. Specifying the dimensions of imports and exports are only required at the root level, though it is recommended that you specify them for clarity.


Any behavior not mentioned here is undefined.

## Supported Types

The only currently supported type is `float32`.

## Supported ONNX OpTypes

The currently supported op types are:


| ONNX OpType | Tested ONNX Compile Support        | Tested Training Support           |
| ----------- | ---------------------------------- |-----------------------------------|
| Add         | <span style=color:green>Yes</span> | <span style=color:green>Yes</span>|
| Concat      | <span style=color:green>Yes</span> | <span style=color:green>Yes</span>|
| Identity    | <span style=color:green>Yes</span> | <span style=color:green>Yes</span>|
| LeakyRelu   | <span style=color:green>Yes</span> | <span style=color:green>Yes</span>|
| LpNormalization | <span style=color:green>Yes</span> | <span style=color:green>Yes</span>|
| MatMul      | <span style=color:green>Yes</span> | <span style=color:green>Yes</span>|
| Mul         | <span style=color:green>Yes</span> |<span style=color:green>Yes</span>|
| Relu        | <span style=color:green>Yes</span> |<span style=color:green>Yes</span>|
| ReduceMean  | <span style=color:green>Yes</span> |<span style=color:green>Yes</span>|
| Softmax     | <span style=color:green>Yes</span> |<span style=color:green>Yes</span>|
| Sqrt        | <span style=color:green>Yes</span> |<span style=color:green>Yes</span>|
| Sub         | <span style=color:green>Yes</span> |<span style=color:green>Yes</span>|
| Transpose   | <span style=color:green>Yes</span> |<span style=color:green>Yes</span>|

Additional notes on functionality that might differ from ONNX. For most details, see [the Onnx documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md).

| ONNX OpType | Notes                                |
| ----------- | -------------------------------------|
| Transpose   | Important difference with the Onnx documentation: by default, when imported into PyTorch, the transpose operator will keep the first dimension the same so as to support batching. The Onnx default behavior is to reverse all the dimensions. |

## Syntax Highlighting in VSCode

If you'd like to use the extension `.agr` for clarity, you can enable syntax highlighting in vscode by placing the following in a settings.json file:
```
"files.associations": {
    "*.agr": "xml"
}
```
To create that settings file, use the command pallet (CTRL-SHIFT-P), type `settings.json`, and choose the appropriate option.

# Training

ONNX is built for inference. However, various supporters of ONNX, including Microsoft (via the onnxruntime), have tried to implement training support. As far as I can tell, Microsoft gave up on trying to support training ONNX files directly. Many of the training tools in onnxruntime are either experimental or scheduled to be depricated. What they did end up implementing was a tool to train PyTorch models (i.e. objects of classes that inherit from torch.nn.Module). Their tool is more narrowly for speeding up training that you could already do natively in PyTorch, and it is not used in this project.

Another option, besides trying to rely on existing ONNX training projects, would have been to make our own. It is actually relatively straightforward: the ONNX file itself is a highly expressive computational graph. We could build a separate graph for training, which has gradient nodes added. It could even take parameters as input and output new parameters while keeping all the data on a GPU. The key is having access to (or building from scratch) nodes that can compute the gradient of each operation (there are many, but they are relatively simple). I ultimately decided (like Microsoft) that this was not worth the pain.

Instead, we opt for converting onnx files to PyTorch. We provide utilities to do that and to use the training features of PyTorch.

Unfortunately, PyTorch does not natively support importing ONNX files. But there is a work-around: building on top of some community tools, we can make our own ONNX to PyTorch converter that is suitable for training. There is more information in the README.md under src/agrippa/onnx2torch for details on exactly how a particular community project was modified. It does not support all ONNX operations, but neither does our markup language.

## Usage

The following code snippet takes a project directory, converts it to an onnx file, then uses the build-in ONNX-to-PyTorch converter to create a PyTorch model, which can be trained in the usual way.

```
import agrippa

proj_name = 'simple-project'
onnx_out = 'simple_testing.onnx'

agrippa.export(proj_name, onnx_out)

torch_model = agrippa.onnx_to_torch(onnx_out)
```

# Utilities

Some utilities are available in agrippa.utils. This includes `find_params`, which returns weight (parameter) names and values as a dictionary. It also includes `save_torch_model`, which takes trained weights from a PyTorch model and saves them into a fresh `weights.pkl` file.

## Finding Parameters

The returned dictionary includes parameters whose names contain the `name` argument (first argument) as a substring. Searching for weights in this way is recommended, since the names of parameters might be changed when the markup is compiled (for example, the names of weights that appear in repeated blocks). The `find_params` function takes two mandatory parameters and one optional: the substring that will be matched (mandatory), the directory of the project (mandatory), and the path to the weights file name within that directory (optional).

### Example

```
matches = agrippa.utils.find_params('bias', 'FNN')
print(matches)
```
The above code might print:
```
{'biases$1': array([[-0.77192398],
       [-0.02351803],
       [-0.00533084],
       [ 0.13640493],
       [-0.12087004]]), 'biases$2': array([[-0.18979854],
       [-0.15769928],
       [ 0.46656397],
       [-0.10602235]])}
```

## Saving Model from PyTorch

After importing your model to PyTorch using `agrippa.onnx_to_torch`, you probably would like to save the trained weights. When imported into PyTorch, the names of the weights change slightly, so it is recommended that you save your models using `agrippa.utils.save_torch_model`, which takes as parameters the PyTorch model, the project directory, and (optionally) the weights filename inside that directory. Under the hood, this function loops over the `state_dict` of the PyTorch model, removes `initializer.` from the parameter's name, and saves it inside a dictionary to `weights.pkl`.

### Example

```
# ... training loop
agrippa.utils.save_torch_model(torch_model, "my-project", "weights.pkl")
```

# Examples

The following architecture is a simple feed forward network with five layers followed by a normalization. The architecture is organized into two blocks, the FFN and the norm layer. Inside the FFN is a FFN Layer block, which is repeated five times.

```
<model script-version="0.0.1">

    <import dim="[3, 1]" from="input" type="float32" />

    <block title="FFN">
        <import from="input" />
        <block title="FFN Layer" rep="5">
            <import from="input" />
            <node title="Linear" op="MatMul">
                <params dim="[3, 3]" name="W" type="float32" />
                <input dim="[3, 1]" src="input" />
                <output dim="[3, 1]" name="linear" />
            </node>
            <node title="Bias" op="Add">
                <params dim="[3, 1]" name="B" type="float32" />
                <input dim="[3, 1]" src="linear" />
                <output dim="[3, 1]" name="biased" />
            </node>
            <node title="ReLu" op="Relu">
                <input dim="[3, 1]" src="biased" />
                <output dim="[3, 1]" name="relu" />
            </node>
            <export from="relu" />
        </block>
    </block>
    <block title="Norm">
        <import from="relu" />
        <node title="ReLu" op="LpNormalization" axis="0" p="1">
            <input dim="[3, 1]" src="relu" />
            <output dim="[3, 1]" name="y" />
        </node>
        <export from="y" />
    </block>
 
    <export dim="[3, 1]" from="y" type="float32" />

</model>
```

You can find more example projects inside the `tests` folder.
