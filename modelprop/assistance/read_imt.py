#!/usr/bin/env python3

import argparse
import json

def main():
    help_text='''
    Prints all of the imt values from the taxonomy functions.
    '''
    argparser = argparse.ArgumentParser(description=help_text)

    argparser.add_argument(
        'fragilityfile',
        help='Filename for the fragility functions'
    )
    
    args = argparser.parse_args()

    with open(args.fragilityfile, 'rt') as input_file:
        fragility = json.load(input_file)

    imts = set()

    for taxonomy_dataset in fragility['data']:
        imts.add(taxonomy_dataset['imt'])
    
    print(sorted(imts))

if __name__ == '__main__':
    main()
