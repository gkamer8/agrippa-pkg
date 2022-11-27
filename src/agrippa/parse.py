from math import exp
from xml.dom import SyntaxErr
import xml.etree.ElementTree as ET
import onnx
import json
import numpy as np
import os
import pickle
from agrippa.expr_parser import parse_expr

WEIGHTS_FNAME = "weights.pkl"
META_FNAME = "meta.json"

ONNX_TYPE_DICT = {
    "float32": onnx.TensorProto.FLOAT
}

DEFAULT_TYPE = "float32"

suppress_prints = False

# For print statements
def _notify(str):
    global suppress_prints
    if not suppress_prints:
        print(str)

# Variables in the file need to be of the form var(myVarName)
def _resolve_attr(text, bindings, expect_value=True):
    if bindings:
        for key in bindings:
            text = text.replace("var("+str(key)+")", str(bindings[key]))
    # detect unbound variable
    if text.find("var(") != -1:
        raise SyntaxWarning(f"Unbound variable found in attribute '{text}'")
    
    # If there's an expression, parse it
    """
    CHANGE THIS CODE TO:
    - delinate the beginning and end of expr( and )
    - delinate the beginning and end of the actual inner expression (simple offsets)
    - make a "to replace" to call on the original text
    """
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

def _resolve_param(name, data_type, dims, weights, init_type="normal", init_args=None):
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

    # Uniformly random between [-1, 1)

    if init_type == 'normal':
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
        bindings=None
    ):

    global suppress_prints
    suppress_prints = suppress
    

    def get_output_model_name():
        if outfile:
            return outfile
        return infile + ".onnx"

    proj_dir = os.listdir(infile)
    arch_file = None
    for fil in proj_dir:
        try:
            if fil[-4:] == ".xml" or fil[-4:] == ".agr":
                arch_file = fil
                break
        except:
            pass 

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

    # the reason why we made those block ids
    saved_import_names = {}  # block id -> dict of names to restore

    name_resolves = {}  # an important data structure for dealing with repetitions and the resulting name changes

    # How many programming sins can we commit in the shortest amount of code?
    # A hellish mix of state and functional programming.
    def parse_block(block, block_id, rep_index=0):

        # Go through each node in the block
        for node in block.findall('node'):
            try:
                op = _resolve_attr(node.attrib['op'], bindings, expect_value=False)
            except KeyError:
                raise SyntaxError("Op type not specified for some node")

            try:
                title = node.attrib['title']
                title = _resolve_attr(node.attrib['title'], bindings, expect_value=False)
            except KeyError:
                title = "GenericNode"  # node title is optional
            title = get_unique_node_name(title)
            
            # Get inputs and outputs
            input_els = node.findall("input")
            naive_inputs = [_resolve_attr(el.attrib['src'], bindings, expect_value=False) for el in input_els]
            inputs = [name_resolves[x] if x in name_resolves else x for x in naive_inputs]

            output_els = node.findall("output")
            naive_outputs = [_resolve_attr(el.attrib['name'], bindings, expect_value=False) for el in output_els]
            outputs = [name_resolves[x] if x in name_resolves else x for x in naive_outputs]

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
                    init_type = "normal"
                try:
                    init_args = _resolve_attr(param.attrib['init_args'], bindings, expect_value=True)
                except:
                    init_args = None
                
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
                    _notify("No keepdims attr selected for ReduceMean; default=1.")
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

            new_node = onnx.helper.make_node(
                name=title,
                op_type=op,
                inputs=inputs,
                outputs=outputs,
                **kwargs
            )
            all_nodes.append(new_node)
        
        # child blocks
        for new_block in block.findall('block'):
            parse_block(new_block, make_unique_block_id())
        
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
                try:
                    saved_import_names[block_id][name] = name_resolves[name]
                except KeyError:
                    saved_import_names[block_id][name] = name
        
        rep_index += 1
        if rep_index < rep:
            curr_exports = []
            # Find the current exports so that we can import them next time
            for exp in block.findall('export'):
                if _resolve_attr(exp.attrib['from'], bindings, expect_value=False) in name_resolves:
                    curr_exports.append(name_resolves[_resolve_attr(exp.attrib['from'], bindings, expect_value=False)])
                else:
                    curr_exports.append(exp.attrib['from'])

            # change the intermediate node resolves as though they were exports
            # this is OK I think; treating exports separately is still good to line up rep imports/exports
            # Note that iter gets nested children as well
            for node in block.iter('node'):
                for out_el in node.findall("output"):
                    name = _resolve_attr(out_el.attrib['name'], bindings, expect_value=False)
                    uni = get_unique_name(name)
                    name_resolves[name] = uni  # for next rep
            
            # Order matters!
            # change the import resolves
            for i, imp in enumerate(block.findall('import')):
                try:
                    name_resolves[_resolve_attr(imp.attrib['from'], bindings, expect_value=False)] = curr_exports[i]
                except IndexError:  # it's ok if some imports come from somewhere else and aren't repeated (they must be the last ones)
                    break

            parse_block(block, block_id, rep_index=rep_index)
        else:
            # Restore import name resolves and leave
            for name in saved_import_names[block_id]:
                name_resolves[name] = saved_import_names[block_id][name]


    # Go through each block (does this recursively)
    for block in root.findall('block'):
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

    onnx.save(model_def, get_output_model_name())

    # Write the weights file if we're supposed to
    if write_weights:
        weights_path = os.path.join(infile, WEIGHTS_FNAME)
        with open(weights_path, "wb") as fhand:
            pickle.dump(weights, fhand)
