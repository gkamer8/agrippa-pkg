import json
from lib2to3.pgen2 import token

# OP: PRECEDENCE
BIN_OP_PRECEDENCE = {
    '^': 2,
    '*': 1,
    '/': 1,
    '//': 1,
    '%': 1,
    '-': 0,
    '+': 0,
}

BIN_OP_FUNCTIONS = {
    '^': lambda x, y: x**y,
    '*': lambda x, y: x*y,
    '/': lambda x, y: x/y,
    '//': lambda x, y: x // y,
    '%': lambda x, y: x%y,
    '-': lambda x, y: x-y,
    '+': lambda x, y: x+y
}

# For dealing with two char long ops
# suffix -> full op
BIN_OP_SUFFIXES = {
    '/': '//'
}

def _tokenize(text, bindings):
    tokens = []

    working_text = ""
    for c in text:
        if c in BIN_OP_PRECEDENCE:
            if working_text:
                tokens.append(working_text)
                working_text = ""
            if c in BIN_OP_SUFFIXES:
                if len(tokens) > 0:
                    working_op = tokens[-1] + c
                    if working_op in BIN_OP_PRECEDENCE:
                        tokens[-1] = working_op
                        continue
            tokens.append(c)
        elif c in [' ']:  # separation characters
            if working_text:
                tokens.append(working_text)
                working_text = ""
        else:
            working_text += c
    if working_text:
        tokens.append(working_text)

    # Apply bindings
    tokens = [str(bindings[x]) if x in bindings else x for x in tokens]
    return tokens

def _has_leq_precedence(first, second):
    return BIN_OP_PRECEDENCE[first] <= BIN_OP_PRECEDENCE[second]

def _infix_to_prefix(infix):
    operators = []
    operands = []
 
    for i in range(len(infix)):
        if infix[i] not in BIN_OP_FUNCTIONS:
            operands.append(infix[i])
        else:
            while len(operators)!=0 and _has_leq_precedence(infix[i], operators[-1]):
                op1 = operands[-1]
                operands.pop()
 
                op2 = operands[-1]
                operands.pop()
 
                op = operators[-1]
                operators.pop()
 
                tmp = [op] + [op2] + [op1]
                operands.append(tmp)
            operators.append(infix[i])
 
    while (len(operators)!=0):
        op1 = operands[-1]
        operands.pop()
 
        op2 = operands[-1]
        operands.pop()
 
        op = operators[-1]
        operators.pop()
 
        tmp = [op] + [op2] + [op1]
        operands.append(tmp)

    return operands[-1]

def _eval(toks):

    # Sometimes this function gets a singleton expression (string)
    if type(toks) != type([]):
        return json.loads(toks)

    toks = toks[::-1]
    stack = []
    for tok in toks:
        if tok in BIN_OP_PRECEDENCE:
            operand1 = stack.pop()
            operand2 = stack.pop()
            # If we're operating on them, they must be values of some kind
            # Kind of hacky
            operand1 = json.loads(str(operand1))
            operand2 = json.loads(str(operand2))

            func = BIN_OP_FUNCTIONS[tok]
            try:
                app = func(operand1, operand2)
            except:
                raise SyntaxError(f"Incompatible operands {operand1}, {operand2} for operator {tok}")
            stack.append(app)
        else:
            stack.append(tok)
    if len(stack) == 0:
        raise SyntaxError("Invalid expression")
    return stack[0]

def parse_expr(text, bindings={}):

    if '(' in text or ')' in text:
        raise SyntaxWarning("Parentheses not supported; ignoring.")

    tokens = _tokenize(text, bindings)

    prefixed = _infix_to_prefix(tokens)

    evaluated = _eval(prefixed)

    return evaluated
