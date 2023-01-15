from math import exp
from xml.dom import SyntaxErr
import xml.etree.ElementTree as ET
import onnx
import json
import numpy as np
import os
import pickle
from agrippa.expr_parser import parse_expr
import math

WEIGHTS_FNAME = "weights.pkl"
META_FNAME = "meta.json"

ONNX_TYPE_DICT = {
    "float32": onnx.TensorProto.FLOAT
}

DEFAULT_TYPE = "float32"
DEFAULT_INIT_TYPE = "xavier"

suppress_prints = False  # global variable

# For print statements
def _notify(str):
    global suppress_prints
    if not suppress_prints:
        print(str)

"""
Gets raw attribute text, a dictionary of global variable bindings
Returns the true attribute value, with variables and expressions resolved
expect_value is True if the return value is not meant to be a string
"""
def _resolve_attr(text, bindings, expect_value=True):
    if bindings:
        for key in bindings:
            text = text.replace("var("+str(key)+")", str(bindings[key]))
    # detect unbound variable
    if text.find("var(") != -1:
        raise SyntaxWarning(f"Unbound variable found in attribute '{text}'")
    
    # If there's an expression, parse it
    to_replace = {}
    total_offset = 0
    remaining = text[total_offset:]
    while remaining.find('expr(') != -1:
        begin_call = text[total_offset:].find('expr(')
        start_i = begin_call + len("expr(")
        
        expr = ""
        end_place = -1
        parens = []
        for i, c in enumerate(remaining[start_i:]):
            if c == ')':
                if len(parens) == 0:
                    end_place = i + start_i
                    break
                else:
                    parens.pop()
            elif c == '(':
                parens.append('(')
            else:
                expr += c
        
        if end_place == -1:
            raise SyntaxError(f"Mismatched parenthesis in expression '{text}'")
        if not bindings:
            bindings = {}

        parsed_expr = parse_expr(expr, bindings=bindings)
        end_call = end_place + len(")")

        to_replace[remaining[begin_call:end_call]] = str(parsed_expr)

        total_offset += end_call
        remaining = text[total_offset:]

    # Substitute the expressions for their values
    for key in to_replace:
        text = text.replace(key, to_replace[key])

    if expect_value:
        return json.loads(text)
    return text


"""
Searches in the weights file for a corresponding weight value or creates a new initializer
Options for init_type are:
- normal, uni_random, ones, zeros
- init_args takes parameterizations of those initialization types
See the README for more info
"""
def _resolve_param(name, data_type, dims, weights, init_type=DEFAULT_INIT_TYPE, init_args=None):
    if name in weights:
        try:
            tens = weights[name]
            res = onnx.helper.make_tensor(
                name=name,
                data_type=data_type,
                dims=dims,
                vals=tens.flatten().tolist())
            return res
        except:
            _notify(f"Error loading weight {name}; arbitrarily initializing.")

    _notify(f"Weight {name} not found; initializing")
    if init_type == 'xavier':
        # Get the sum of the last two dimensions
        # If there is only one, use just that one
        stdev = 1
        if len(dims) >= 2:
            stdev = math.sqrt(2. / (dims[-1] + dims[-2]))
        elif len(dims) == 1:
            stdev = 1. * math.sqrt(dims[0])
        tens = np.random.normal(loc=0., scale=stdev, size=dims)
    elif init_type == 'normal':
        mu = 0
        sigma = 1
        if init_args is not None:
            if len(init_args) < 2:
                _notify(f"Init args for normal initialization has too few members ({len(init_args)} < 2); using mu=0, std=1")
            else:
                mu = init_args[0]
                sigma = init_args[1]
        tens = np.random.normal(loc=mu, scale=sigma, size=dims)
    elif init_type == 'uni_random':
        a = -1
        b = 1
        if init_args is not None:
            if len(init_args) < 2:
                _notify(f"Init args for uniform random initialization has too few members ({len(init_args)} < 2); using a=-1, b=1")
            else:
                a = init_args[0]
                b = init_args[1]
        tens = np.random.random(dims) * (b - a) + a
    elif init_type == 'ones':
        tens = np.ones(dims)
    elif init_type == 'zeros':
        tens = np.zeros(dims)
    elif init_type == 'constant':
        if len(init_args) != 1:
            raise SyntaxError(f"For init type = constant, init args must have exactly one member")
        val = init_args[0]
        tens = np.full(dims, val)
    else:
        _notify(f"Weight initialization type '{init_type}' not known; defaulting to N(0, 1)")
        tens = np.random.normal(loc=0., scale=1., size=dims)
    res = onnx.helper.make_tensor(
            name=name,
            data_type=data_type,
            dims=dims,
            vals=tens.flatten().tolist())

    weights[name] = tens  # weights is an object, so this is allowed
    return res

# Determines the output file name of the onnx model
# outfile = what the user wants it named
# infile = the project folder name, which is used as a default (with .onnx)
def _get_output_model_name(outfile, infile):
    if outfile:
        return outfile
    return infile + ".onnx"


"""
def export(infile, ...)

This function is used to convert a project directory into an ONNX file.
The infile parameter should be a directory.
The outfile parameter defaults to your project name with a .onnx extension.

"""
def export(
        infile,
        outfile=None,
        producer="Unknown",
        graph_name="Unknown",
        write_weights=True,
        suppress=False,
        reinit=False,
        bindings=None,
        index=None  # Main file that things are imported into
    ):

    global suppress_prints
    suppress_prints = suppress

    proj_dir = os.listdir(infile)
    arch_file = None
    if index is None:
        for fil in proj_dir:
            try:
                if fil[-4:] == ".xml" or fil[-4:] == ".agr":
                    arch_file = fil
                    break
            except:
                pass
    else:
        arch_file = index

    if not arch_file:
        raise FileNotFoundError("No model architecture file found. A file with .agr or .xml extension expected.")

    # Look for a meta file
    meta_file = None
    for fil in proj_dir:
        try:
            if fil == META_FNAME:
                meta_file = fil
                break
        except:
            pass
    if not meta_file:
        _notify("No meta file found - using default metadata.")
    else:
        with open(os.path.join(infile, meta_file)) as fhand:
            meta = json.load(fhand)
            try:
                producer = meta['producer']
            except KeyError:
                _notify("Meta file does not contain producer - using default.")
            try:
                graph_name = meta['graph_name']
            except KeyError:
                _notify("Meta file does not contain graph name - using default.")

    # Look for a weights file
    weights = {}
    weights_file = None
    for fil in proj_dir:
        try:
            if fil == WEIGHTS_FNAME:
                weights_file = fil
                break
        except:
            pass
    if not weights_file:
        _notify("No weights file found - using arbitrary initialization.")
    elif reinit:
        _notify("Re-initializing weights, as per your request.")
    else:
        with open(os.path.join(infile, weights_file), "rb") as fhand:
            weights = pickle.load(fhand)

    tree = ET.parse(os.path.join(infile, arch_file))  # should throw error if not well-formed XML
    root = tree.getroot()

    # Get root imports/exports

    all_nodes = []
    all_inputs = []
    all_outputs = []
    all_inits = []

    # Root model imports
    for child in root.findall("import"):

        child_type = DEFAULT_TYPE
        if 'type' in child.attrib:
            child_type = child.attrib['type']

        name = _resolve_attr(child.attrib['from'], bindings, expect_value=False)
        child_type = _resolve_attr(child_type, bindings, expect_value=False)
        dim = _resolve_attr(child.attrib['dim'], bindings, expect_value=True)
        x = onnx.helper.make_tensor_value_info(name,
                                            ONNX_TYPE_DICT[child_type],
                                            dim)
        all_inputs.append(x)

    parameter_repeats = {}  # for shared params, decide whether to create new weight value for a parameter
    def get_unique_param_name(name):
        if name in parameter_repeats:
            parameter_repeats[name] += 1
            return name + "$" + str(parameter_repeats[name])
        parameter_repeats[name] = 1
        return name + "$1"

    # used to make sure we get a unique weight name + we can handle nested repeats
    # must return a different name from name even if it doesn't exist yet in the repeats dictionary
    repeats = {}
    def get_unique_name(name):
        if name in repeats:
            repeats[name] += 1
            return name + "$" + str(repeats[name])
        repeats[name] = 1
        return name + "$1"

    repeat_node_names = {}
    def get_unique_node_name(name):
        if name in repeat_node_names:
            repeat_node_names[name] += 1
            return name + "$" + str(repeat_node_names[name])
        repeat_node_names[name] = 1
        return name + "$1"

    block_id_tracker = {'curr': 0}  # A hack to not technically use a global var
    def make_unique_block_id():
        block_id_tracker['curr'] += 1
        return block_id_tracker['curr']

    def make_imported_name(name, imp_name):
        if imp_name is not None:
            return imp_name + "$" + name
        return name

    # the reason why we made those block ids
    saved_import_names = {}  # block id -> dict of names to restore

    name_resolves = {}  # an important data structure for dealing with repetitions and the resulting name changes

    saved_to_concat_exports = {}  # block id -> dict orig export name -> lists of export names to concat

    # How many programming sins can we commit in the shortest amount of code?
    # A hellish mix of state and functional programming.
    # "imp_name" means import name = i.e., name needed to resolve names in another file
    def parse_block(block, block_id, imp_name=None, rep_index=0, stretch_index=0):

        # Go through each child of the block
        # Could be block or node
        for node in block.findall('*'):

            # Processing children in correct order w.r.t. nodes so that topological sort works
            if node.tag == 'block':
                parse_block(node, make_unique_block_id(), imp_name=imp_name)
                continue
            elif node.tag != 'node':
                continue

            try:
                op = _resolve_attr(node.attrib['op'], bindings, expect_value=False)
            except KeyError:
                raise SyntaxError("Op type not specified for some node")

            try:
                title = node.attrib['title']
                title = _resolve_attr(node.attrib['title'], bindings, expect_value=False)
            except KeyError:
                title = "GenericNode"  # node title is optional
            title = make_imported_name(title, imp_name)
            title = get_unique_node_name(title)
            
            # Get inputs and outputs
            input_els = node.findall("input")
            naive_inputs = [_resolve_attr(el.attrib['src'], bindings, expect_value=False) for el in input_els]
            inputs = [make_imported_name(x, imp_name) for x in naive_inputs]  # change name if file is from other source
            inputs = [name_resolves[x] if x in name_resolves else x for x in inputs]

            output_els = node.findall("output")
            naive_outputs = [_resolve_attr(el.attrib['name'], bindings, expect_value=False) for el in output_els]
            outputs = [make_imported_name(x, imp_name) for x in naive_outputs]  # change name if file is from other source
            outputs = [name_resolves[x] if x in name_resolves else x for x in outputs]

            # Make the parameters
            params = []
            param_precedence = []
            all_children = node.findall("*")
            precendence_counter = 0
            for el in all_children:
                if el.tag == "params":
                    param = el
                    param_precedence.append(precendence_counter)
                    precendence_counter += 1
                elif el.tag == "input":
                    precendence_counter += 1
                    continue
                else:
                    continue
                # Note: order can be important!
                # Each op type specifies the order in which inputs are meant to be included
                orig_name = _resolve_attr(param.attrib['name'], bindings, expect_value=False)
                if imp_name is not None:
                    orig_name = make_imported_name(orig_name, imp_name)
                name = orig_name

                try:
                    shared = _resolve_attr(param.attrib['shared'], bindings, expect_value=False)
                except:
                    shared = "no"
                
                dim = _resolve_attr(param.attrib['dim'], bindings, expect_value=True)
                
                param_type = DEFAULT_TYPE
                if 'type' in param.attrib:
                    param_type = param.attrib['type']

                dtype = _resolve_attr(param_type, bindings, expect_value=False)

                try:
                    init_type = _resolve_attr(param.attrib['init'], bindings, expect_value=False)
                except:
                    init_type = DEFAULT_INIT_TYPE
                try:
                    init_args = _resolve_attr(param.attrib['init_args'], bindings, expect_value=True)
                except:
                    init_args = None

                orig_name = _resolve_attr(param.attrib['name'], bindings, expect_value=False)

                # If the parameter is frozen, add "$constant" to the name
                try:
                    frozen = _resolve_attr(param.attrib['frozen'], bindings, expect_value=False)
                except:
                    frozen = "no"
                if frozen == "yes":
                    orig_name += "$constant"

                name = orig_name
                
                if shared == "no":
                    name = get_unique_param_name(orig_name)
                    param_onnx = _resolve_param(name, ONNX_TYPE_DICT[dtype], dim, weights, init_type=init_type, init_args=init_args)
                    all_inits.append(param_onnx)
                else:
                    # First time we see it, even if we keep the name, need to initialize
                    if name not in parameter_repeats:
                        param_onnx = _resolve_param(name, ONNX_TYPE_DICT[dtype], dim, weights, init_type=init_type, init_args=init_args)
                        all_inits.append(param_onnx)
                        parameter_repeats[name] = 1  # so we don't do this next time
                params.append(name)

            kwargs = {}

            def splice_inputs(inputs, params, param_precedence):
                to_return = [None for _ in range(len(inputs) + len(params))]
                input_index = 0
                param_index = 0
                for i in range(len(to_return)):
                    if param_index < len(params) and param_precedence[param_index] == i:
                        to_return[i] = params[param_index]
                        param_index += 1
                    elif input_index < len(inputs):
                        to_return[i] = inputs[input_index]
                        input_index += 1
                return to_return
            # Sometimes there are other tags defining certain attributes
            # Here is where we would need to support new op types
            if op == 'MatMul':
                inputs = splice_inputs(inputs, params, param_precedence)
            elif op == "Add":
                inputs = splice_inputs(inputs, params, param_precedence)
            elif op == "Relu":
                inputs = splice_inputs(inputs, params, param_precedence)
            elif op == "LpNormalization":
                kwargs["axis"] = _resolve_attr(node.attrib['axis'], bindings, expect_value=True)
                kwargs["p"] = _resolve_attr(node.attrib['p'], bindings, expect_value=True)
                inputs = splice_inputs(inputs, params, param_precedence)
                if len(inputs) > 1:
                    raise SyntaxWarning(f"LpNormalization should only have one input, yet it has {len(inputs)}")
            elif op == "Transpose":
                try:
                    perm = node.attrib['perm']
                    kwargs["perm"] = _resolve_attr(perm, bindings, expect_value=True)
                except KeyError:
                    pass  # Default perm value, which is reverse
                inputs = splice_inputs(inputs, params, param_precedence)
                if len(inputs) > 1:
                    raise SyntaxWarning(f"Transpose should only have one input, yet it has {len(inputs)}")
            elif op == "Identity":
                inputs = splice_inputs(inputs, params, param_precedence)
                if len(inputs) > 1:
                    raise SyntaxWarning(f"Identity should only have one input, yet it has {len(inputs)}")
            elif op =="Softmax":
                kwargs["axis"] = _resolve_attr(node.attrib['axis'], bindings, expect_value=True)
                inputs = splice_inputs(inputs, params, param_precedence)
                if len(inputs) > 1:
                    raise SyntaxWarning(f"Softmax should only have one input, yet it has {len(inputs)}")
            elif op == "Div":
                inputs = splice_inputs(inputs, params, param_precedence)
                if len(inputs) != 2:
                    raise SyntaxWarning(f"Div should only have exactly 2 inputs, yet it has {len(inputs)}")
            elif op == "Mul":
                inputs = splice_inputs(inputs, params, param_precedence)
                if len(inputs) != 2:
                    raise SyntaxWarning(f"Mul should have exactly 2 inputs, yet it has {len(inputs)}")
            elif op == "ReduceMean":
                kwargs["axes"] = _resolve_attr(node.attrib['axes'], bindings, expect_value=True)
                try:
                    kwargs["keepdims"] = _resolve_attr(node.attrib['keepdims'], bindings, expect_value=True)
                except KeyError:
                    # Default is 1 (used to be a notify - don't put a notify here, it's jank)
                    pass
                inputs = splice_inputs(inputs, params, param_precedence)
                if len(inputs) > 1:
                    raise SyntaxWarning(f"Softmax should only have one input, yet it has {len(inputs)}")
            elif op == "Sub":
                inputs = splice_inputs(inputs, params, param_precedence)
                if len(inputs) != 2:
                    raise SyntaxWarning(f"Sub should have exactly 2 inputs, yet it has {len(inputs)}")
            elif op == "Sqrt":
                inputs = splice_inputs(inputs, params, param_precedence)
                if len(inputs) != 1:
                    raise SyntaxWarning(f"Sqrt should only have 1 input, yet it has {len(inputs)}")
            elif op == "Concat":
                inputs = splice_inputs(inputs, params, param_precedence)
                try:
                    kwargs["axis"] = _resolve_attr(node.attrib['axis'], bindings, expect_value=True)
                except KeyError:
                    raise SyntaxError(f"Expecting attribute 'axis' for Concat node")
            elif op == "LeakyRelu":
                inputs = splice_inputs(inputs, params, param_precedence)
                try:
                    kwargs["alpha"] = _resolve_attr(node.attrib['alpha'], bindings, expect_value=True)
                except KeyError:
                    raise SyntaxError(f"Expecting attribute 'alpha' for LeakyRelu node")
            new_node = onnx.helper.make_node(
                name=title,
                op_type=op,
                inputs=inputs,
                outputs=outputs,
                **kwargs
            )
            all_nodes.append(new_node)
        
        
        """
        Stretch:
        
        Change intermediate node output names, but keep
            import names
        Save new export names for later
        Call parse block until stretch_index runs out
        If we're the final one, create a new node that concats all the saved export names, and
            this node should have output names associated with the original exports
        Can't stretch with rep, so just return from function

        """

        try:
            stretch = _resolve_attr(block.attrib['stretch'], bindings)
        except KeyError:
            stretch = 1
        
        if stretch > 1:

            curr_exports = []

            # MIGHT NEED TO CHANGE SO I CAN ACCESS INTERMEDIATE NAMES

            # save the export resolves
            for exp in block.findall('export'):
                name = _resolve_attr(exp.attrib['from'], bindings, expect_value=False)
                if imp_name is not None:
                    name = make_imported_name(name, imp_name)
                if block_id in saved_to_concat_exports:
                    try:
                        saved_to_concat_exports[block_id][name].append(name_resolves[name])
                    except KeyError:
                        saved_to_concat_exports[block_id][name].append(name)
                else:
                    try:
                        saved_to_concat_exports[block_id] = {name: [name_resolves[name]]}
                    except KeyError:
                        saved_to_concat_exports[block_id] = {name: [name]}

            if stretch_index < stretch - 1:
                # change the intermediate node resolves
                # Note that iter gets nested children as well
                for node in block.iter('node'):
                    for out_el in node.findall("output"):
                        name = _resolve_attr(out_el.attrib['name'], bindings, expect_value=False)
                        if imp_name is not None:
                            name = make_imported_name(name, imp_name)
                        uni = get_unique_name(name)
                        name_resolves[name] = uni  # for next rep
                stretch_index += 1
                parse_block(block, block_id, imp_name=imp_name, rep_index=rep_index, stretch_index=stretch_index)
            else:
                # Create concat nodes
                for orig_name in saved_to_concat_exports[block_id]:                
                    concat_inputs = saved_to_concat_exports[block_id][orig_name]
                    new_name = orig_name + "$concat"
                    new_name = get_unique_name(new_name)
                    kwargs = {"axis": -1}  # will probably have to change, or provide options
                    concat_node = onnx.helper.make_node(
                        name=make_imported_name(get_unique_node_name("stretch_concat"), imp_name),
                        op_type="Concat",
                        inputs=concat_inputs,
                        outputs=[new_name],
                        **kwargs
                    )
                    all_nodes.append(concat_node)
                    name_resolves[orig_name] = new_name

        # Forget reps if we're stretching
        if stretch > 1:
            return

        """
        
        Now dealing with reps

        """

        # Does the block have a rep?
        try:
            rep = _resolve_attr(block.attrib['rep'], bindings)
        except KeyError:
            rep = 1

        # Save the name state of your imports so that you can return them when you leave
        if rep_index == 0:
            saved_import_names[block_id] = {}
            for imp in block.findall('import'):
                name = imp.attrib['from']
                if imp_name is not None:
                    name = make_imported_name(name, imp_name)
                try:
                    saved_import_names[block_id][name] = name_resolves[name]
                except KeyError:
                    saved_import_names[block_id][name] = name
        
        rep_index += 1
        if rep_index < rep:
            curr_exports = []
            # Find the current exports so that we can import them next time
            for exp in block.findall('export'):
                bad_name = _resolve_attr(exp.attrib['from'], bindings, expect_value=False)
                good_name = make_imported_name(bad_name, imp_name)
                if good_name in name_resolves:
                    curr_exports.append(name_resolves[good_name])
                else:
                    curr_exports.append(make_imported_name(exp.attrib['from'], imp_name))

            # change the intermediate node resolves as though they were exports
            # this is OK I think; treating exports separately is still good to line up rep imports/exports
            # Note that iter gets nested children as well
            for node in block.iter('node'):
                for out_el in node.findall("output"):
                    bad_name = _resolve_attr(out_el.attrib['name'], bindings, expect_value=False)
                    good_name = make_imported_name(bad_name, imp_name)
                    uni = get_unique_name(good_name)
                    name_resolves[good_name] = uni  # for next rep
            
            # Order matters!
            # change the import resolves
            for i, imp in enumerate(block.findall('import')):
                try:
                    bad_name = _resolve_attr(imp.attrib['from'], bindings, expect_value=False)
                    good_name = make_imported_name(bad_name, imp_name)
                    name_resolves[good_name] = curr_exports[i]
                except IndexError:  # it's ok if some imports come from somewhere else and aren't repeated (they must be the last ones)
                    break

            parse_block(block, block_id, imp_name=imp_name, rep_index=rep_index)
        else:
            # Restore import name resolves and leave
            for name in saved_import_names[block_id]:
                name_resolves[name] = saved_import_names[block_id][name]


    # Go through each block (does this recursively)
    for block in root.findall('block'):
        # Is block imported?
        if 'src' in block.attrib:
            try:
                imp_name = block.attrib['name']
            except KeyError:
                raise SyntaxError(f"An imported block ({block.attrib['src']}) is missing 'name' attribute")

            print(block.attrib['src'])
            # Go out and find this jawns
            imp_tree = ET.parse(os.path.join(infile, block.attrib['src']))  # should throw error if not well-formed XML
            imp_block = imp_tree.getroot()
            parse_block(imp_block, make_unique_block_id(), imp_name=imp_name)
        else:
            parse_block(block, make_unique_block_id())

    # Root model exports - their names might have changed!
    for child in root.findall("export"):

        child_type = DEFAULT_TYPE
        if 'type' in child.attrib:
            child_type = child.attrib['type']

        naive_name = _resolve_attr(child.attrib['from'], bindings, expect_value=False)

        if naive_name in name_resolves:
            name = name_resolves[naive_name]
        else:
            name = naive_name

        child_type = _resolve_attr(child_type, bindings, expect_value=False)
        dim = _resolve_attr(child.attrib['dim'], bindings, expect_value=True)
        x = onnx.helper.make_tensor_value_info(name,
                                            ONNX_TYPE_DICT[child_type],
                                            dim)
        all_outputs.append(x)

    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=all_nodes,
        name=graph_name,
        inputs=all_inputs,  # Graph input
        outputs=all_outputs,  # Graph output
        initializer=all_inits,
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name=producer)
    model_def.opset_import[0].version = 13

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    onnx.save(model_def, _get_output_model_name(outfile, infile))

    # Write the weights file if we're supposed to
    if write_weights:
        weights_path = os.path.join(infile, WEIGHTS_FNAME)
        with open(weights_path, "wb") as fhand:
            pickle.dump(weights, fhand)
