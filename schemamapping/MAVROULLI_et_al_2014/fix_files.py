#!/usr/bin/env python3

import glob
import json
import os

def fix_data(data):
    old_source_name = data['source_name']
    source_schema, source_taxonomy = old_source_name.split('-')

    old_target_name = data['target_name']
    target_schema, target_taxonomy = old_target_name.split('-')

    old_conv_matrix = data['conv_matrix']

    conv_matrix = json.loads(old_conv_matrix)

    source_damage_states = data['source_damage_states']
    target_damage_states = data['target_damage_states']

    return {
        'source_schema': source_schema,
        'source_taxonomy': source_taxonomy,
        'target_schema': target_schema,
        'target_taxonomy': target_taxonomy,
        'source_damage_states': source_damage_states,
        'target_damage_states': target_damage_states,
        'conv_matrix': conv_matrix,
    }

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    all_json_files = glob.glob(os.path.join(current_dir, '*.json'))
    
    for json_file in all_json_files:
        basename = os.path.basename(json_file)
        with open(json_file, 'rt') as input_file:
            data = json.load(input_file)
            fixed_data = fix_data(data)
            output_path = os.path.join(current_dir, 'output', basename)

            with open(output_path, 'wt') as output_file:
                json.dump(fixed_data, output_file)

if __name__ == '__main__':
    main()
