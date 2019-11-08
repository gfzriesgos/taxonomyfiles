#!/usr/bin/env python3

"""
This script is to add the schema to a replacement cost
file that just contains the data-array.

Prints on stdout, so you can redirect it to a file
using the shell.
"""

import argparse
import json
import sys

def main():
    arg_parser = argparse.ArgumentParser(
        'Adds the meta section to the file'
    )
    arg_parser.add_argument(
        'repl_cost_file',
        help='The file to load for the replacment costs'
    )
    arg_parser.add_argument(
        'schema',
        help='The schema to add.'
    )

    args = arg_parser.parse_args()

    with open(args.repl_cost_file, 'rt') as input_file:
        data = json.load(input_file)

        output = {
            'meta': {
                'id': args.schema
            },
            'data': data
        }
        json.dump(output, sys.stdout, indent=4)

if __name__ == '__main__':
    main()

        

