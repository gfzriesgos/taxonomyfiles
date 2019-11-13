#!/usr/bin/env python3

"""
This is a script to fix the taxonomy schema mapping file
that Camilo sent me.

It should care about the following tasks:
- the source_schema must be torres et. al
- The target schema must be mavrouli et. el
- The source taxonomies must replace the '-' with an '_'.
- The matrix must be transposed, so that the top level keys
  are the source taxonomies.
"""

import json
import pandas

def fix_source_taxonomy(tax):
    return tax.replace('-', '_')

def main():
    input_file_name = 'Conversion_Matrix_Torres-Corredor-et-al-2017_Mavrouli-et-al-2014_completed.json'


    NAME_TORRES = 'Torres_Corredor_et_al_2017'
    NAME_MAVROULI = 'Mavrouli_et_al_2014'
    with open(input_file_name, 'rt') as input_file:
        data = json.load(input_file)

    data['source_schema'] = NAME_TORRES
    data['target_schema'] = NAME_MAVROULI

    data['source_taxonomies'] = [
        fix_source_taxonomy(x)
        for x in data['source_taxonomies']
    ]
    conv_matrix = pandas.DataFrame(data['conv_matrix']).transpose()
    conv_matrix.columns = [
        fix_source_taxonomy(x)
        for x in conv_matrix.columns
    ]
    data['conv_matrix'] = conv_matrix.to_dict()

    print(json.dumps(data, indent=4))



if __name__ == '__main__':
    main()
