#!/usr/bin/env python3

'''
The idea of this test case is to
handle all the schemas and taxonomies
consistent, so that they can run on deus
in later stages.

Just run the test with python3 test_all.py
'''

import collections
import glob
import json
import os
import unittest

# This should be the names for the schemas
NAME_SARA = 'SARA_v1.0'
NAME_HAZUS = 'HAZUS_v1.0'
NAME_SUPPASRI = 'SUPPASRI2013_v2.0' 
# not all schemas are supported for all # data at the moment
# for full support all schemas should be included
# in all kind of files
# TODO: All should be fully supported!!!!
SCHEMAS_WITH_EXPOSURE = [NAME_SARA]
SCHEMAS_WITH_FRAGILITY = [NAME_SARA, NAME_HAZUS, NAME_SUPPASRI]
SCHEMAS_WITH_MAPPING = [NAME_SARA, NAME_SUPPASRI]
SCHEMAS_WITH_REPLACEMENT_COSTS = [NAME_SARA, NAME_SUPPASRI]

class TestAll(unittest.TestCase):
    '''
    Test case to test all the constraints we need
    to make the files and the services consistent.
    '''

    def setUp(self):
        '''
        Setup method, so that all the test cases
        have the following datasets.
        '''
        self.fragility_data = FragilityData.read_from_json_files()
        self.exposure_data = ExposureData.read_from_json_files()
        self.replacementcost_data = ReplacementcostData.read_from_json_files()
        self.schemamapping_data = SchemaMappingData.read_from_json_files()

    def test_exposure(self):
        '''
        All the expected schemas should be included
        for the exposure.
        '''
        for schema in SCHEMAS_WITH_EXPOSURE:
            self.assertIn(schema, self.exposure_data.get_schemas())

    def test_fragility_models(self):
        '''
        We also need the fragility models for the expected schemas.
        '''
        for schema in SCHEMAS_WITH_FRAGILITY:
            self.assertIn(schema, self.fragility_data.get_schemas())

    def test_schemamappings(self):
        '''
        We need the schema mappings.
        '''
        for schema in SCHEMAS_WITH_MAPPING:
            self.assertIn(schema, self.schemamapping_data.get_source_schemas())

    def test_replacement_costs(self):
        '''
        We need the supported schemas.
        '''
        for schema in SCHEMAS_WITH_REPLACEMENT_COSTS:
            self.assertIn(schema, self.replacementcost_data.get_schemas())

    def test_schema_tax_from_exposure_is_in_fragility_meta(self):
        '''
        All the taxonomies from the exposure model (assetmaster)
        should be included in the the fragility functions (in their meta section).
        '''
        for schema in SCHEMAS_WITH_EXPOSURE:
            tax_exposure = set(self.exposure_data.get_taxonomies(schema))
            tax_fragility = set(self.fragility_data.get_taxonomies_in_meta(schema))

            for tax in tax_exposure:
                self.assertIn(tax, tax_fragility)

    def test_schema_tax_from_fragility_meta_is_in_exposure(self):
        '''
        Also for all the schemas for which we can provide
        the exposure and the fragility models,
        we want to make sure that the taxonomies from
        the fragility models are included in the exposure model.
        '''
        for schema in set(SCHEMAS_WITH_EXPOSURE) & set(SCHEMAS_WITH_MAPPING):
            tax_exposure = set(self.exposure_data.get_taxonomies(schema))
            tax_fragility = set(self.fragility_data.get_taxonomies_in_meta(schema))

            for tax in tax_fragility:
                self.assertIn(tax, tax_exposure)

    def test_schema_tax_from_fragility_meta_is_in_fragility_data(self):
        '''
        For all the fragility models we want to make sure that
        all the taxonomies in the meta section are the same as
        those in the data section.
        '''

        schemas_from_fragility = self.fragility_data.get_schemas()

        for schema in schemas_from_fragility:
            tax1_fragility = set(self.fragility_data.get_taxonomies_in_meta(schema))
            tax2_fragility = set(self.fragility_data.get_taxonomies_in_data(schema))

            for tax in tax1_fragility:
                self.assertIn(tax, tax2_fragility)

            for tax in tax2_fragility:
                self.assertIn(tax, tax1_fragility)

    def test_schema_tax_from_fragility_meta_have_schema_mapping(self):
        '''
        We also want to make sure that all the taxomomies
        that we have for fragility and mapping match.
        '''

        schemas_by_fragility = self.fragility_data.get_schemas()
        source_schemas_by_schemamappings = self.schemamapping_data.get_source_schemas()

        for schema in set(SCHEMAS_WITH_MAPPING) & set(SCHEMAS_WITH_FRAGILITY):
            self.assertIn(schema, source_schemas_by_schemamappings)
            self.assertIn(schema, schemas_by_fragility)

            tax_fragility = self.fragility_data.get_taxonomies_in_meta(schema)
            tax_schema_mapping = self.schemamapping_data.get_source_taxonomies(schema)

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

        schemas_by_fragiltiy = self.fragility_data.get_schemas()
        target_schemas_by_schemamappings = self.schemamapping_data.get_target_schemas()

        for target_schema in target_schemas_by_schemamappings:
            self.assertIn(target_schema, schemas_by_fragiltiy)

            tax_fragility = self.fragility_data.get_taxonomies_in_meta(target_schema)

            for target_taxonomy in self.schemamapping_data.get_target_taxonomies(target_schema):
                self.assertIn(target_taxonomy, tax_fragility)

    def test_no_duplicates_in_exposure_json_taxonomies(self):
        '''
        Tests if there are duplicates in each single taxonomy list
        in the exposure json files.
        '''

        schemas_by_exposure = self.exposure_data.get_schemas()

        for schema in schemas_by_exposure:
            tax_list = self.exposure_data.get_taxonomies(schema)
            tax_set = set(tax_list)

            self.assertEqual(len(tax_list), len(tax_set))

    def test_tax_in_fragility_models_are_there_for_replacement_costs(self):
        '''
        Tests that the taxonomies in the fragility model are all
        also in the replacement costs (as as we support as
        fragility models, we want to compute the loss in deus as well.)
        '''

        schemas_by_repl_costs = self.replacementcost_data.get_schemas()
        schemas_by_fragility = self.fragility_data.get_schemas()

        for schema in set(schemas_by_repl_costs) & set(schemas_by_fragility):
            tax_by_repl_costs = self.replacementcost_data.get_taxonomies(schema)
            tax_by_fragility = self.fragility_data.get_taxonomies_in_meta(schema)

            for tax in tax_by_fragility:
                self.assertIn(tax, tax_by_repl_costs)

            for tax in tax_by_repl_costs:
                self.assertIn(tax, tax_by_fragility)
    
    def test_schema_mappings_sum_to_1(self):
        '''
        All the schema mappings from one taxonomy in one
        schema to the target taxonomy should sum to 1
        as we don't want to lose or add buildings
        on mapping to a different schema.
        '''

        InputSetting = collections.namedtuple(
            'InputSetting',
            'source_schema source_tax target_schema ds'
        )

        sum_by_input_setting = collections.defaultdict(lambda: 0)

        for dataset in self.schemamapping_data.data:
            source_schema = dataset['source_schema']
            source_tax = dataset['source_taxonomy']
            target_schema = dataset['target_schema']
            target_tax = dataset['target_taxonomy']
            conv_matrix = dataset['conv_matrix']
            source_damage_states = conv_matrix.keys()

            for damage_state in source_damage_states:
                parts_in_target_damage_states = conv_matrix[damage_state].values()
                input_setting = InputSetting(
                    source_schema=source_schema,
                    source_tax=source_tax,
                    target_schema=target_schema,
                    ds=damage_state
                )

                sum_by_input_setting[input_setting] += sum(parts_in_target_damage_states)

        for input_setting in sum_by_input_setting.keys():
            self.assertLess(0.95, sum_by_input_setting[input_setting])
            self.assertLess(sum_by_input_setting[input_setting], 1.05)

    def test_structure_for_schema_mappings(self):
        '''
        This is the test for the schema mapping files, so that
        the source damage states are in the keys of the conv matrix,
        while the target damage states are the keys of those values.

        For example:

        "source_damage_states": [0, 1, 2, 3, 4],
        "target_damage_states": [0, 1, 2, 3, 4, 5, 6],
        "conv_matrix": {
            "0": {
                "0": 0.9999753112,
                "1": 7.7378e-06,
                "2": 0.0787385548,
                "3": 0.0328946953,
                "4": 0.0276871971,
                "5": 0.0019610316,
                "6": 0.0723992114
            },
        ...

        The keys in conv_matrix are the source_damage_states,
        while the keys in the inner dict are the target_damage_states.
        '''

        for dataset in self.schemamapping_data.data:
            source_damage_states = dataset['source_damage_states']
            target_damage_states = dataset['target_damage_states']

            conv_matrix = dataset['conv_matrix']

            keys_conv_matrix = conv_matrix

            for source_damage_state in source_damage_states:
                self.assertIn(str(source_damage_states), keys_conv_matrix)

                for target_damage_state in target_damage_states:
                    self.assertIn(str(target_damage_states), keys_conv_matrix[str(source_damage_states)].keys())

    def test_imt_and_imu_values(self):
        '''
        Tests the imt and imu values.
        '''

        imt_and_imu_to_accept = {
            NAME_SARA: {
                'PGA': 'g',
                'SA(0.1)': 'g',
                'SA(0.3)': 'g',
            },
            NAME_SUPPASRI: {
                'ID': 'm'
            }
        }

        for schema in imt_and_imu_to_accept.keys():
            self.assertIn(schema, self.fragility_data.get_schemas())
            imt_and_imu_values = self.fragility_data.get_imt_and_imu_values(schema)

            for imt in imt_and_imu_values.keys():
                self.assertEqual(1, len(imt_and_imu_values[imt]))
                imu = list(imt_and_imu_values[imt])[0]

                imt_to_check = imt.upper()
                self.assertIn(imt_to_check, imt_and_imu_to_accept[schema].keys())
                self.assertEqual(imu, imt_and_imu_to_accept[schema][imt_to_check])


class FragilityData():
    def __init__(self, data):
        self.data = data

    @classmethod
    def read_from_json_files(cls):
        fragility_models_by_schema = {}

        current_dir = os.path.dirname(os.path.abspath(__file__))
        modelprop_dir = os.path.join(current_dir, 'modelprop')
        json_files = glob.glob(os.path.join(modelprop_dir, '*.json'))

        for json_file in json_files:
            with open(json_file, 'rt') as input_file:
                data = json.load(input_file)
                schema = data['meta']['id']

                fragility_models_by_schema[schema] = data
        return cls(fragility_models_by_schema)

    def get_schemas(self):
        return self.data.keys()

    def get_taxonomies_in_meta(self, schema):
        return self.data[schema]['meta']['taxonomies']

    def get_taxonomies_in_data(self, schema):
        inner_data = self.data[schema]['data']
        results = []
        for dataset in inner_data:
            tax = dataset['taxonomy']
            results.append(tax)
        return results

    def get_imt_and_imu_values(self, schema):
        inner_data = self.data[schema]['data']

        results = collections.defaultdict(set)

        for dataset in inner_data:
            imt = dataset['imt']
            imu = dataset['imu']

            results[imt].add(imu)

        return results


class ExposureData():
    def __init__(self, data):
        self.data = data

    @classmethod
    def read_from_json_files(cls):
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
        return cls(taxonomies_by_schema)

    def get_schemas(self):
        return self.data.keys()

    def get_taxonomies(self, schema):
        return self.data[schema]

class SchemaMappingData():
    def __init__(self, data):
        self.data = data

    @classmethod
    def read_from_json_files(cls):
        datasets = []

        current_dir = os.path.dirname(os.path.abspath(__file__))
        schemamapping_dir = os.path.join(current_dir, 'schemamapping')
        json_files = glob.glob(os.path.join(schemamapping_dir, '*', '*.json'))

        for json_file in json_files:
            with open(json_file, 'rt') as input_file:
                data = json.load(input_file)
                datasets.append(data)

        return cls(datasets)

    def get_source_schemas(self):
        result = set()

        for dataset in self.data:
            source_schema = dataset['source_schema']
            result.add(source_schema)

        return result

    def get_source_taxonomies(self, schema):
        result = set()

        for dataset in self.data:
            source_schema = dataset['source_schema']
            if schema == source_schema:
                source_taxonomy = dataset['source_taxonomy']
                result.add(source_taxonomy)
        return result

    def get_target_schemas(self):
        result = set()
        for dataset in self.data:
            target_schema = dataset['target_schema']
            result.add(target_schema)
        return result

    def get_target_taxonomies(self, schema):
        result = set()
        for dataset in self.data:
            target_schema = dataset['target_schema']
            if schema == target_schema:
                target_taxonomy = dataset['target_taxonomy']
                result.add(target_taxonomy)
        return result

class ReplacementcostData():
    def __init__(self, data):
        self.data = data

    @classmethod
    def read_from_json_files(cls):
        replacement_costs_by_schema = {}

        current_dir = os.path.dirname(os.path.abspath(__file__))
        replacementcost_dir = os.path.join(current_dir, 'replacement_costs')
        json_files = glob.glob(os.path.join(replacementcost_dir, '*.json'))

        for json_file in json_files:
            with open(json_file, 'rt') as input_file:
                data = json.load(input_file)

                schema = data['meta']['id']

                replacement_costs_by_schema[schema] = data

        return cls(replacement_costs_by_schema)

    def get_schemas(self):
        return self.data.keys()

    def get_taxonomies(self, schema):
        result = set()
        for dataset in self.data[schema]['data']:
            tax = dataset['taxonomy']
            result.add(tax)
        return result


if __name__ == '__main__':
    unittest.main()
