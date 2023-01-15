# TODOs

A list of what's being worked on and what needs to be done.

## Documentation
- Give details on every onnx command - note changed behavior of transpose operation w.r.t. pytorch batching
- Document expressions in attributes
- Concat operator
- LeakyRelu operator
- Importing from other files

## Implementation

### Minor
- Create "default" tag to set default initializations, other things
- types of initializations and their implementations need to be separated out for maintainability
- Aliases for operations (e.g., "relu" for "Relu" - should go with clean ups to parse)

### Major
- Figure out batching in ONNX
- "stretch" attribute for multi headed attention
- Clean up parse code, namespace situation
- Better XML parse errors
- Better error handling w.r.t. shape problems
- Allow for defining constants as well as parameters
- Define parameter as frozen/constant, and provide frozen/constant value (+ make weight naming/file transparent) + will probably have to make a bindings file

## Testing

### Examples
- Transformer decoder - delineate d_key, d_query, d_value
- Use expression to determine weight 

### Unit tests
- More thorough testing of expr (empty expressions, unbound variables, etc.)
