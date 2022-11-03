import xml.etree.ElementTree as ET
import onnx
import json
import numpy as np


def export(infile, outfile=None, producer="Unknown", graph_name="Unknown"):

    ONNX_TYPE_DICT = {
        "float32": onnx.TensorProto.FLOAT
    }

    def get_producer_name():
        return producer

    def get_graph_name():
        return graph_name

    def get_output_model_name():
        if outfile:
            return outfile
        return infile + ".onnx"

    # In the future, should look for a weights file
    def resolve_param(name, data_type, dims):

        dims = json.loads(dims)

        # For now, just do initalization of 1s
        tens = np.ones(dims)

        res = onnx.helper.make_tensor(
                name=name,
                data_type=data_type,
                dims=dims,
                vals=tens.flatten().tolist())
        return res

    tree = ET.parse(infile)
    root = tree.getroot()

    # Get root imports/exports

    all_nodes = []
    all_inputs = []
    all_outputs = []
    all_inits = []

    # Root model imports
    for child in root.findall("import"):
        name = child.attrib['from']
        child_type = child.attrib['type']
        dim = child.attrib['dim']
        dim = json.loads(dim)
        x = onnx.helper.make_tensor_value_info(name,
                                            ONNX_TYPE_DICT[child_type],
                                            dim)
        all_inputs.append(x)

    # Root model exports
    for child in root.findall("export"):
        name = child.attrib['from']
        child_type = child.attrib['type']
        dim = child.attrib['dim']
        dim = json.loads(dim)
        x = onnx.helper.make_tensor_value_info(name,
                                            ONNX_TYPE_DICT[child_type],
                                            dim)
        all_outputs.append(x)

    parameter_repeats = {}  # for shared params, decide whether to create new weight value for a parameter

    # used to make sure we get a unique weight name + we can handle nested repeats
    # must return a different name from name even if it doesn't exist yet in the repeats dictionary
    repeats = {}
    def get_unique_name(name):
        if name in repeats:
            repeats[name] += 1
            return name + str(repeats[name])
        repeats[name] = 1
        return name + str(repeats[name])

    repeat_node_names = {}
    def get_unique_node_name(name):
        if name in repeat_node_names:
            repeat_node_names[name] += 1
            return name + str(repeat_node_names[name])
        repeat_node_names[name] = 1
        return name + str(repeat_node_names[name])

    block_id_tracker = {'curr': 0}

    def make_unique_block_id():
        block_id_tracker['curr'] += 1
        return block_id_tracker['curr']

    # the reason why we made those block ids
    saved_import_names = {}  # block id -> dict of names to restore

    name_resolves = {}  # an important data structure for dealing with repetitions and the resulting name changes

    def parse_block(block, block_id, rep_index=0):

        # Go through each node in the block
        for node in block.findall('node'):
            op = node.attrib['op']
            title = node.attrib['title']
            title = get_unique_node_name(title)
            
            # Get inputs and outputs
            input_els = node.findall("input")
            naive_inputs = [el.attrib['src'] for el in input_els]
            inputs = [name_resolves[x] if x in name_resolves else x for x in naive_inputs]

            output_els = node.findall("output")
            naive_outputs = [el.attrib['name'] for el in output_els]
            outputs = [name_resolves[x] if x in name_resolves else x for x in naive_outputs]

            # Make the parameters
            params = []
            param_els = node.findall("params")
            for param in param_els:
                # Note: order can be important!
                # Each op type specifies the order in which inputs are meant to be included
                name = param.attrib['name']
                if True:  # TODO: deal with weight sharing
                    name = get_unique_name(name)
                params.append(name)

                dim = param.attrib['dim']
                dtype = param.attrib['type']
                if name not in parameter_repeats:  # What if we're (not) sharing a parameter?
                    param_onnx = resolve_param(name, ONNX_TYPE_DICT[dtype], dim)
                    all_inits.append(param_onnx)
                    parameter_repeats[name] = 1

            kwargs = {}
            # Sometimes there are other tags defining certain attributes
            # Here is where we would need to support new op types
            if op == 'MatMul':
                inputs = params + inputs
            if op == "Add":
                inputs = params + inputs
            if op == "Relu":
                pass
            if op == "LpNormalization":
                kwargs["axis"] = int(node.attrib['axis'])  # -1 means last; note that in a 2d system, 0 is row wise, 1 is column wise
                kwargs["p"] = int(node.attrib['p'])  # I think it's L1 norm?


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
            rep = int(block.attrib['rep'])
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
                if exp.attrib['from'] in name_resolves:
                    curr_exports.append(name_resolves[exp.attrib['from']])
                else:
                    curr_exports.append(exp.attrib['from'])

            # change the intermediate node resolves as though they were exports
            # this is OK I think; treating exports separately is still good to line up rep imports/exports
            # Note that iter gets nested children as well
            for node in block.iter('node'):
                for out_el in node.findall("output"):
                    name = out_el.attrib['name']
                    uni = get_unique_name(name)
                    name_resolves[name] = uni  # for next rep
            
            # Order matters!
            # change the import resolves
            for i, imp in enumerate(block.findall('import')):
                try:
                    name_resolves[imp.attrib['from']] = curr_exports[i]
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

    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=all_nodes,
        name=get_graph_name(),
        inputs=all_inputs,  # Graph input
        outputs=all_outputs,  # Graph output
        initializer=all_inits,
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name=get_producer_name())
    model_def.opset_import[0].version = 13

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    onnx.save(model_def, get_output_model_name())