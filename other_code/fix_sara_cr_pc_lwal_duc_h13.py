#!/usr/bin/env python3

"""
The aim of this script is to fix all the schema mapping files
that reference to the sara schema and the CR-PC-LWAL-DUC-H1-3
taxonomy.

They should be replaced by the ones for CR-PC-LWAL-H1-3.
"""

import glob
import json
import os

NAME_SARA = 'SARA_v1.0'
TAX_TO_FIX = 'CR-PC-LWAL-DUC-H1-3'
TAX_FIXED = 'CR-PC-LWAL-H1-3'

def read_json(filename):
    with open(filename, 'rt') as input_file:
        return json.load(input_file)

def need_fix(data):
    if data['source_schema'] == NAME_SARA:
        return data['source_taxonomy'] == TAX_TO_FIX
    if data['target_schema'] == NAME_SARA:
        return data['target_taxonomy'] == TAX_TO_FIX
    return False

def fix(data):
    if data['source_schema'] == NAME_SARA:
        if data['source_taxonomy'] == TAX_TO_FIX:
            data['source_taxonomy'] = TAX_FIXED
    if data['target_schema'] == NAME_SARA:
        if data['target_taxonomy'] == TAX_TO_FIX:
            data['target_taxonomy'] = TAX_FIXED
    return data

def write_json(data, filename):
    with open(filename, 'wt') as output_file:
        json.dump(data, output_file)

def main():
    for filename in glob.glob('*.json'):
        data = read_json(filename)
        if need_fix(data):
            fixed = fix(data)
            output_path = os.path.join(
                'fixed',
                os.path.basename(filename),
            )
            write_json(fixed, output_path)



if __name__ == '__main__':
    main()
