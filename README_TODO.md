# TODOs

A list of what's being worked on and what needs to be done.

## Documentation
- Nothing

## Implementation

### Minor
- Create "default" tag to set default initializations, other things
- types of initializations and their implementations need to be separated out for maintainability
- Make sure rep as attribute of imported file block works properly.
- Aliases for operations (e.g., "relu" for "Relu" - should go with clean ups to parse)

### Major
- Figure out batching in ONNX
- Clean up parse code, namespace situation
- Better XML parse errors
- Better error handling w.r.t. shape problems

## Testing

### Examples
- Use constant/frozen features to simplify transformer

### Unit tests
- More thorough testing of expr (empty expressions, unbound variables, etc.)
