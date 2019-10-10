#!/usr/bin/env python3

'''
The idea of this test case is to
handle all the schemas and taxonomies
consistent, so that they can run on deus
in later stages.

Just run the test with python3 test_all.py
'''

import glob
import json
import os
import unittest

# This should be the names for the schemas
NAME_SARA = 'SARA_v1.0'
NAME_HAZUS = 'HAZUS_v1.0'
NAME_SUPPASRI = 'SUPPASRI2013_v2.0'

# not all schemas are supported for all
# data at the moment
SCHEMAS_WITH_EXPOSURE = [NAME_SARA]
SCHEMAS_WITH_FRAGILITY = [NAME_SARA, NAME_HAZUS, NAME_SUPPASRI]
SCHEMAS_WITH_MAPPING = [NAME_SARA, NAME_SUPPASRI]

class TestAll(unittest.TestCase):
    '''
    Test case to test all the constraints we need
    to make the files and the services consistent.
    '''

    def test_exposure(self):
        '''
        All the expected schemas should be included
        for the exposure.
        '''
        taxonomies_by_schema = read_exposure_json_files() 
        for schema in SCHEMAS_WITH_EXPOSURE:
            self.assertIn(schema, taxonomies_by_schema.keys())

    def test_fragility_models(self):
        '''
        We also need the fragility models for the expected schemas.
        '''
        fragility_models_by_schema = read_fragility_json_files()
        for schema in SCHEMAS_WITH_FRAGILITY:
            self.assertIn(schema, fragility_models_by_schema.keys())

    def test_schemamappings(self):
        '''
        We need the schema mappings.
        '''
        schema_mappings_by_schema = read_schema_mappings()
        for schema in SCHEMAS_WITH_MAPPING:
            self.assertIn(schema, schema_mappings_by_schema.keys())
        

    def test_schema_tax_from_exposure_is_in_fragility_meta(self):
        '''
        All the taxonomies from the exposure model (assetmaster)
        should be included in the the fragility functions (in their meta section).
        '''

        taxonomies_by_schema = read_exposure_json_files() 
        fragility_models_by_schema = read_fragility_json_files()

        for schema in SCHEMAS_WITH_EXPOSURE:
            tax_exposure = set(taxonomies_by_schema[schema])
            data_fragility = fragility_models_by_schema[schema]
            tax1_fragility = set(data_fragility['meta']['taxonomies'])

            for tax in tax_exposure:
                self.assertIn(tax, tax1_fragility)

    def test_schema_tax_from_fragility_meta_is_in_exposure(self):
        '''
        Also for all the schemas for which we can provide
        the exposure and the fragility models,
        we want to make sure that the taxonomies from
        the fragility models are included in the exposure model.
        '''
        taxonomies_by_schema = read_exposure_json_files() 
        fragility_models_by_schema = read_fragility_json_files()

        for schema in set(SCHEMAS_WITH_EXPOSURE) & set(SCHEMAS_WITH_MAPPING):
            tax_exposure = set(taxonomies_by_schema[schema])
            data_fragility = fragility_models_by_schema[schema]
            tax1_fragility = set(data_fragility['meta']['taxonomies'])

            for tax in tax1_fragility:
                self.assertIn(tax, tax_exposure)

    def test_schema_tax_from_fragility_meta_is_in_fragility_data(self):
        '''
        For all the fragility models we want to make sure that
        all the taxonomies in the meta section are the same as
        those in the data section.
        '''

        fragility_models_by_schema = read_fragility_json_files()

        for key in fragility_models_by_schema.keys():
            data_fragility = fragility_models_by_schema[key]
            tax1_fragility = set(data_fragility['meta']['taxonomies'])
            tax2_fragility = set(x['taxonomy'] for x in data_fragility['data'])

            for tax in tax1_fragility:
                self.assertIn(tax, tax2_fragility)

            for tax in tax2_fragility:
                self.assertIn(tax, tax1_fragility)

    def test_schema_tax_from_fragility_meta_have_schema_mapping(self):
        '''
        We also want to make sure that all the taxomomies
        that we have for fragility and mapping match.
        '''

        fragility_models_by_schema = read_fragility_json_files()
        schema_mappings = read_schema_mappings()

        for schema in set(SCHEMAS_WITH_MAPPING) & set(SCHEMAS_WITH_FRAGILITY):
            self.assertIn(schema, schema_mappings.keys())
            self.assertIn(schema, fragility_models_by_schema.keys())

            tax_fragility = set(fragility_models_by_schema[schema]['meta']['taxonomies'])
            tax_schema_mapping = set(x['source_schema'] for x in schema_mappings[schema])

            for tax in tax_fragility:
                self.assertIn(tax, tax_schema_mapping)

    def test_schema_tax_targets_supported_by_fragilities(self):
        '''
        We want to make sure that we support all the
        taxonomies in the fragility functions that
        are the target_schemas and target_taxonomies
        in the schema mappings
        (so that we can run deus again).
        '''

        fragility_models_by_schema = read_fragility_json_files()

        # first get all the target schemas
        target_taxonomies_by_target_schema = {}

        schema_mappings = read_schema_mappings()
        for source_schema in schema_mappings.keys():
            for entry in schema_mappings[source_schema]:
                target_schema = entry['target_schema']
                target_taxonomy = entry['target_taxonomy']

                if not target_schema in target_taxonomies_by_target_schema.keys():
                    target_taxonomies_by_target_schema[target_schema] = set()

                target_taxonomies_by_target_schema[target_schema].add(target_taxonomy)

        for target_schema in target_taxonomies_by_target_schema.keys():
            self.assertIn(target_schema, fragility_models_by_schema.keys())

            data_fragility = fragility_models_by_schema[target_schema]
            tax_fragility = set(data_fragility['meta']['taxonomies'])

            for target_taxonomy in target_taxonomies_by_target_schema[target_schema]:
                self.assertIn(target_taxonomy, tax_fragility)

    def test_no_duplicates_in_exposure_json_taxonomies(self):
        '''
        Tests if there are duplicates in each single taxonomy list
        in the exposure json files.
        '''

        taxonomies_by_schema = read_exposure_json_files() 

        for schema in taxonomies_by_schema.keys():
            tax_list = taxonomies_by_schema[schema]
            tax_set = set(tax_list)

            self.assertEqual(len(tax_list), len(tax_set))


def read_fragility_json_files():
    fragility_models_by_schema = {}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    modelprop_dir = os.path.join(current_dir, 'modelprop')
    json_files = glob.glob(os.path.join(modelprop_dir, '*.json'))

    for json_file in json_files:
        with open(json_file, 'rt') as input_file:
            data = json.load(input_file)
            schema = data['meta']['id']

            fragility_models_by_schema[schema] = data
    return fragility_models_by_schema
    

def read_exposure_json_files():
    taxonomies_by_schema = {}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    assetmaster_dir = os.path.join(current_dir, 'assetmaster')
    json_files = glob.glob(os.path.join(assetmaster_dir, '*.json'))

    for json_file in json_files:
        with open(json_file, 'rt') as input_file:
            data = json.load(input_file)
            schema = data['id']

            taxonomies = data['taxonomies']

            taxonomies_by_schema[schema] = taxonomies
    return taxonomies_by_schema


def read_schema_mappings():
    schema_mappings_by_schema = {}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    schemamapping_dir = os.path.join(current_dir, 'schemamapping')
    json_files = glob.glob(os.path.join(schemamapping_dir, '*', '*.json'))

    for json_file in json_files:
        with open(json_file, 'rt') as input_file:
            data = json.load(input_file)

            schema = data['source_schema']
            if schema not in schema_mappings_by_schema:
                schema_mappings_by_schema[schema] = []

            schema_mappings_by_schema[schema].append(data)
    return schema_mappings_by_schema


if __name__ == '__main__':
    unittest.main()
