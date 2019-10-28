#!/usr/bin/env python3

"""
This script is meant to fix some issues (mostly
source, target schema + taxonomy and filenames),
so that we can work with it.
"""

import glob
import json
import os

NAME_SARA = 'SARA_v1.0'
NAME_SUPPASRI = 'SUPPASRI2013_v2.0'

def read_json(filename):
    with open(filename, 'rt') as input_file:
        return json.load(input_file)

def get_source_schema(conversion, filename):
    if conversion['target_taxonomy'] == 'SARA_UNK':
        return NAME_SUPPASRI
    return NAME_SARA

def get_target_schema(conversion, filename):
    if conversion['target_taxonomy'] == 'SARA_UNK':
        return NAME_SARA
    return NAME_SUPPASRI

def get_source_taxonomy(conversion, filename):
    if conversion['target_taxonomy'] == 'SARA_UNK':
        return conversion['source_taxonomy']
    return 'UNK'

def get_target_taxonomy(conversion, filename):
    if conversion['target_taxonomy'] == 'SARA_UNK':
        return 'UNK'
    return conversion['target_taxonomy']

def fix(conversion, filename):
    # we can stay with the conversion matrix
    conv_matrix = conversion['conv_matrix']
    source_schema = get_source_schema(conversion, filename)
    source_taxonomy = get_source_taxonomy(conversion, filename)
    target_schema = get_target_schema(conversion, filename)
    target_taxonomy = get_target_taxonomy(conversion, filename)
    source_damage_states = conversion['source_damage_states']
    target_damage_states = conversion['target_damage_states']

    return {
        'source_schema': source_schema,
        'source_taxonomy': source_taxonomy,
        'target_schema': target_schema,
        'target_taxonomy': target_taxonomy,
        'source_damage_states': source_damage_states,
        'target_damage_states': target_damage_states,
        'conv_matrix': conv_matrix,
    }


def create_outfilename(fixed):
    return '{0}_{1}to{2}_{3}.json'.format(
        fixed['source_schema'],
        fixed['source_taxonomy'],
        fixed['target_schema'],
        fixed['target_taxonomy']
    )

def write_json(data, filename):
    with open(filename, 'wt') as out_file:
        json.dump(data, out_file)


def main():
    for filename in glob.glob('*.json'):
        conversion = read_json(filename)
        fixed = fix(conversion, filename)
        out_filename = create_outfilename(fixed)
        dir_outfilename = os.path.join('fixed', out_filename)
        write_json(fixed, dir_outfilename)

if __name__ == '__main__':
    main()
