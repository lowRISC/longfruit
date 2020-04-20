#!/usr/bin/env python3

from optparse import OptionParser
import configparser
import os
import sys

path = '~/tools/longfruit/'
path = os.path.expanduser(path)
sys.path.insert(0, path)
import longfruit

def main():
    parser = OptionParser()
    parser.add_option(
        '-s', '--save', dest='save', default=False, action='store_true',
        help='save result'
    )
    parser.add_option(
        '-v', '--verbose', dest='verbose', default=False, action='store_true',
        help='verbose output'
    )
    options, args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(path + 'scenario.ini')
    scn = config['scenario']
    filename = scn['filename']
    arch = scn['arch']
    abi = scn['abi']
    cost1 = int(scn['cost1'])
    cost2 = int(scn['cost2'])
    c1, c2, asm1, asm2 = longfruit.run_test(filename, arch, abi)
    passed = c1 == cost1 and c2 >= cost2
    scn['cost1'] = str(c1)
    scn['cost2'] = str(c2)
    if options.verbose:
        longfruit.write_result(sys.stdout, config, asm1, asm2)
    if options.save:
        txt = filename[:-1] + 'txt'
        longfruit.write_result(open(txt, 'w'), config, asm1, asm2)
    sys.exit(0 if passed else -1)

if __name__ == '__main__':
    main()
