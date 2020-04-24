#!/usr/bin/env python3

from random import choice, choices, expovariate, randint, randrange, shuffle, uniform
import configparser
import os
import re
import string
import subprocess
import sys

def georand(lmb):
    # roughly geometrically distributed
    return int(expovariate(lmb))

def random_id():
    return ''.join(choices(
        string.ascii_uppercase +
        string.ascii_lowercase +
        string.digits, k=8))

def filter_asm(asm):
    for line in asm.splitlines():
        # remove comments
        m = re.match(r'^([^#]*)#(.*)$', line)
        if m:
            line = m.group(1)

        line = line.strip()

        # skip empty lines
        if len(line) == 0:
            continue

        # skip directives
        if line.startswith('.'):
            continue

        # skip labels
        if line.endswith(':'):
            continue

        yield line

def instr_cost(line):
    parts = re.split('[ \t,]', line)
    instr = parts[0]
    cost_load_store = 2
    cost_fpu_load_store = cost_load_store
    cost_alu = 1
    cost_fpu = 1
    cost_branch = 3
    cost_mul = cost_branch
    cost_div = 8    # TODO: clang generates long instr. sequences instead of div
    cost_call = 20  # TODO: builtins
    cost_ret = 2
    cost_ebreak = 0 # should be unreachable if no undef. behavior is generated

    if instr == 'li':
        imm = int(parts[-1])
        if imm >= -2048 and imm < 2048: # addi
            return cost_alu
        elif (imm & 0xFFF) == 0:        # lui
            return cost_alu
        else:                           # lui + addi
            return cost_alu * 2

    return {
        'add': cost_alu,
        'addi': cost_alu,
        'addiw': cost_alu,
        'addw': cost_alu,
        'and': cost_alu,
        'andi': cost_alu,
        'beq': cost_branch,
        'beqz': cost_branch,
        'bge': cost_branch,
        'bgeu': cost_branch,
        'bgez': cost_branch,
        'bgt': cost_branch,
        'bgtu': cost_branch,
        'bgtz': cost_branch,
        'ble': cost_branch,
        'bleu': cost_branch,
        'blez': cost_branch,
        'blt': cost_branch,
        'bltu': cost_branch,
        'bltz': cost_branch,
        'bne': cost_branch,
        'bnez': cost_branch,
        'call': cost_call,
        'div': cost_div,
        'divu': cost_div,
        'divuw': cost_div,
        'divw': cost_div,
        'ebreak': cost_ebreak,
        'fadd.d': cost_fpu,
        'fadd.s': cost_fpu,
        'fcvt.d.l': cost_fpu,
        'fcvt.d.lu': cost_fpu,
        'fcvt.d.s': cost_fpu,
        'fcvt.d.w': cost_fpu,
        'fcvt.d.wu': cost_fpu,
        'fcvt.l.d': cost_fpu,
        'fcvt.l.s': cost_fpu,
        'fcvt.lu.d': cost_fpu,
        'fcvt.lu.s': cost_fpu,
        'fcvt.s.d': cost_fpu,
        'fcvt.s.l': cost_fpu,
        'fcvt.s.lu': cost_fpu,
        'fcvt.s.w': cost_fpu,
        'fcvt.s.wu': cost_fpu,
        'fcvt.w.d': cost_fpu,
        'fcvt.w.s': cost_fpu,
        'fcvt.wu.d': cost_fpu,
        'fcvt.wu.s': cost_fpu,
        'fdiv.d': cost_fpu,
        'fdiv.s': cost_fpu,
        'feq.d': cost_fpu,
        'feq.s': cost_fpu,
        'fge.d': cost_fpu,
        'fge.s': cost_fpu,
        'fgt.d': cost_fpu,
        'fgt.s': cost_fpu,
        'fld': cost_fpu_load_store,
        'fle.d': cost_fpu,
        'fle.s': cost_fpu,
        'flt.d': cost_fpu,
        'flt.s': cost_fpu,
        'flw': cost_fpu_load_store,
        'fmadd.d': cost_fpu,
        'fmadd.s': cost_fpu,
        'fmul.d': cost_fpu,
        'fmul.s': cost_fpu,
        'fmv.d': cost_fpu,
        'fmv.d.x': cost_fpu,
        'fmv.s': cost_fpu,
        'fmv.s.x': cost_fpu,
        'fmv.w.x': cost_fpu,
        'fmv.x.s': cost_fpu,
        'fneg.d': cost_fpu,
        'fneg.s': cost_fpu,
        #'fnmsub.s': cost_fpu, # not found yet
        'fnmsub.d': cost_fpu,
        'fsd': cost_fpu_load_store,
        'fsub.d': cost_fpu,
        'fsub.s': cost_fpu,
        'fsw': cost_fpu_load_store,
        'j': cost_branch,
        'jr': cost_branch,
        'lb': cost_load_store,
        'lbu': cost_load_store,
        'ld': cost_load_store,
        'lh': cost_load_store,
        'lhu': cost_load_store,
        'lui': cost_alu,
        'lw': cost_load_store,
        'lwu': cost_load_store,
        'mul': cost_mul,
        'mulh': cost_mul,
        'mulhu': cost_mul,
        'mulw': cost_mul,
        'mv': cost_alu,
        'neg': cost_alu,
        'negw': cost_alu,
        'nop': cost_alu,
        'not': cost_alu,
        'or': cost_alu,
        'ori': cost_alu,
        'rem': cost_div,
        'remu': cost_div,
        'remuw': cost_div,
        'remw': cost_div,
        'ret': cost_ret,
        'sb': cost_load_store,
        'sd': cost_load_store,
        'seqz': cost_alu,
        'sext': cost_alu,
        'sext.w': cost_alu,
        'sgt': cost_alu,
        'sgtu': cost_alu,
        'sgtz': cost_alu,
        'sh': cost_load_store,
        'sll': cost_alu,
        'slli': cost_alu,
        'slliw': cost_alu,
        'sllw': cost_alu,
        'slt': cost_alu,
        'slti': cost_alu,
        'sltiu': cost_alu,
        'sltu': cost_alu,
        'snez': cost_alu,
        'sra': cost_alu,
        'srai': cost_alu,
        'sraiw': cost_alu,
        'sraw': cost_alu,
        'srl': cost_alu,
        'srli': cost_alu,
        'srliw': cost_alu,
        'srlw': cost_alu,
        'sub': cost_alu,
        'subw': cost_alu,
        'sw': cost_load_store,
        'xor': cost_alu,
        'xori': cost_alu,
    }[instr]

def compile(compiler, arch, abi, filename):
    if compiler == 'gcc':
        prog = 'riscv64-unknown-linux-gnu-gcc'
        opts = []
    elif compiler == 'clang':
        prog = 'clang'
        opts = [
            '-Wno-literal-conversion',
            '-Wno-implicit-int-float-conversion'
        ]
        if arch == 'rv64gc':
            opts.append('--target=riscv64')
        elif arch == 'rv32gc':
            opts.append('--target=riscv32')
        else:
            assert False, 'unsupported arch'
    else:
        assert False, 'unsupported compiler'

    opts = opts + [
        '-Werror=implicit-int',
        '-Wno-tautological-compare',
        '-Wno-overflow',
        '-Wno-constant-conversion',
        '-Wno-unsequenced',
        f'-march={arch}',
        f'-mabi={abi}',
        '-O2', '-S', '-o', '-', filename
    ]

    r = subprocess.check_output([prog] + opts).decode('utf-8')
    return r

def get_cost(asm):
    cost_total = 0
    for line in filter_asm(asm):
        cost = instr_cost(line)
        cost_total = cost_total + cost
    return cost_total

class Context:
    def __init__(self):
        self.var_counter = 0
        self.vars = []

    def gen_var(self, loop_counter = False):
        if not loop_counter:
            v = f'v{self.var_counter}'
        else:
            v = f'i{self.var_counter}'
        self.var_counter = self.var_counter + 1
        self.vars.append(v)
        return v

    def gen_vars(self, num):
        return [self.gen_var() for i in range(randint(1, num))]

    def rand_var(self):
        return choice(self.vars)

    def copy(self):
        ctx = Context()
        ctx.var_counter = self.var_counter
        ctx.vars = self.vars.copy()
        return ctx

def gen_type():
    return choice([
        'char', 'short', 'int', 'long', 'long long',
        'float', 'double'
    ])

def gen_type_integer():
    signed = choice(['signed', 'unsigned'])
    ty = choice(['char', 'short', 'int', 'long', 'long long'])
    return f'{signed} {ty}'

def gen_cast_integer():
    return f'({gen_type_integer()})'

def gen_expr_literal_int_zero():
    return 0

def gen_expr_literal_int_12_bit():
    return randrange(-2048, 2048)

def gen_expr_literal_int_20_bit_up():
    return randrange(0, 2**20) << 12

def gen_expr_literal_int_32_bit():
    return randrange(-2**31, 2**31)

def gen_expr_literal_int_64_bit():
    return randrange(-2**63, 2**63)

def gen_expr_literal_float():
    return uniform(-1_000_000, 1_000_000)

def gen_expr_literal(ctx = None):
    v = choice([
        gen_expr_literal_int_zero,
        gen_expr_literal_int_12_bit,
        gen_expr_literal_int_20_bit_up,
        gen_expr_literal_int_32_bit,
        gen_expr_literal_int_64_bit,
        gen_expr_literal_float,
    ])()
    return v

def gen_expr_var(ctx):
    return ctx.rand_var()

def gen_expr_unary(ctx):
    a = ctx.rand_var()
    op = choice(['-', '~', '!', '++', '--'])
    cast = ''
    if op == '~':
        # must be applied to an integer operand
        cast = gen_cast_integer()
    return f'{op}{cast}{a}'

def gen_expr_binary(ctx):
    a = ctx.rand_var()
    b = ctx.rand_var()
    ops = [
        '^', '&', '|', '<<', '>>',
        '+', '-',
        '*', '/', '%',
        '==', '!=',
        '<', '<=', '>', '>=',
        '&&', '||'
    ]
    op = choice(ops)
    cast1 = ''
    cast2 = ''
    if op in ['^', '&', '|', '%', '<<', '>>']:
        # must be applied to integer operands
        cast1 = gen_cast_integer()
        cast2 = gen_cast_integer()
    return f'{cast1}{a} {op} {cast2}{b}'

def gen_expr_ternary(ctx):
    a = ctx.rand_var()
    b = ctx.rand_var()
    c = ctx.rand_var()
    return f'{a} ? {b} : {c}'

def gen_expr(ctx):
    return choice([
        gen_expr_var,
        gen_expr_literal,
        gen_expr_unary,
        gen_expr_binary,
        gen_expr_ternary,
    ])(ctx)

def gen_stmt_decl(ctx):
    t = gen_type()
    e = gen_expr(ctx)
    v = ctx.gen_var()
    s = f'{t} {v} = {e};'
    return s

def gen_stmt_assign(ctx):
    # avoid assigning to loop counters
    while True:
        v = ctx.rand_var()
        if v[0] != 'i':
            break
    e = gen_expr(ctx)
    return f'{v} = {e};'

def gen_stmt_loop(ctx):
    loop_ctx = ctx.copy()
    t = gen_type_integer()
    i = loop_ctx.gen_var(loop_counter = True)
    end = randrange(1, 5000)
    return (
        f'for({t} {i} = 0; {i} < {end}; ++{i}) {{\n'
        f'{gen_block(loop_ctx)}'
        f'}}'
    )

def gen_stmt(ctx):
    stmt = choice([
        gen_stmt_decl,
        gen_stmt_assign,
        gen_stmt_loop,
    ])(ctx)
    return f'{stmt}\n'

def gen_block(ctx):
    block = ''
    for i in range(georand(0.5)):
        block = block + gen_stmt(ctx)
    return block

def gen_func_args(ctx):
    n = georand(0.2) + 1
    args = [f'{gen_type()} {v}' for v in ctx.gen_vars(n)]
    return ', '.join(args)

def gen_func(ctx):
    return (
        f'{gen_type()} test({gen_func_args(ctx)}) {{\n'
        f'{gen_block(ctx)}'
        f'return {ctx.rand_var()};\n'
        f'}}'
    )

def gen_global(ctx):
    g = ctx.gen_var()
    return f'{gen_type()} {g} = {gen_expr_literal()};'

def gen_globals(ctx):
    globals = ''
    for i in range(georand(1.0)):
        g = gen_global(ctx)
        globals = f'{globals}{g}\n'
    return globals

def gen_unit(ctx):
    # for now, one function with some parameter and access to some globals
    unit = gen_globals(ctx)
    unit = f'{unit}{gen_func(ctx)}\n'
    return unit

def gen_test(filename):
    with open(filename, 'w') as f:
        ctx = Context()
        print(gen_unit(ctx), file=f)

def test_file(filename, arch, abi):
    asm_gcc = compile('gcc', arch, abi, filename)
    c1 = get_cost(asm_gcc)
    asm_clang = compile('clang', arch, abi, filename)
    c2 = get_cost(asm_clang)
    return c1, c2, asm_gcc, asm_clang

def read_file(fn):
    with open(fn) as f:
        return f.read()

def write_config(filename, arch, abi, cost1, cost2):
    config = configparser.ConfigParser()
    config.add_section('scenario')
    config['scenario']['filename'] = filename
    config['scenario']['arch'] = str(arch)
    config['scenario']['abi'] = str(abi)
    config['scenario']['cost1'] = str(cost1)
    config['scenario']['cost2'] = str(cost2)
    with open('scenario.ini', 'w') as f:
        config.write(f)
    return config

def write_result(f, config, asm1, asm2):
    config.write(f)
    filename = config['scenario']['filename']
    print(f'### Source:\n{read_file(filename)}', file=f)
    print(f'### GCC:\n{asm1}', file=f)
    print(f'### Clang:\n{asm2}', file=f)

def run_test(filename, arch, abi):
    asm1 = compile('gcc', arch, abi, filename)
    c1 = get_cost(asm1)
    asm2 = compile('clang', arch, abi, filename)
    c2 = get_cost(asm2)
    return c1, c2, asm1, asm2

def reduce_case(filename):
    subprocess.check_output(['creduce', 'test.py', filename])

def main():
    while True:
        id = random_id()
        source_file = f'case-{id}.c'
        case_file = f'case-{id}.txt'
        gen_test(source_file)
        scenarios = [
            ['rv32gc', 'ilp32d'],
            ['rv64gc', 'lp64d'],
        ]
        shuffle(scenarios)
        passed = False
        for arch, abi in scenarios:
            c1, c2, asm1, asm2 = run_test(source_file, arch, abi)
            print(c1, c2)
            if c2 > c1:
                passed = True
                config = write_config(source_file, arch, abi, c1, c2)
                print('reducing')
                reduce_case(source_file)
                c1, c2, asm1, asm2 = run_test(source_file, arch, abi)
                write_result(sys.stdout, config, asm1, asm2)
                write_result(open(case_file, 'w'), config, asm1, asm2)
                break
        if not passed:
            os.remove(source_file)

if __name__ == '__main__':
    main()
