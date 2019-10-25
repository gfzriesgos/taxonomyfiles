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

import geopandas as gpd
import pandas as pd

# This should be the names for the schemas
NAME_SARA = 'SARA_v1.0'
NAME_HAZUS = 'HAZUS_v1.0'
NAME_SUPPASRI = 'SUPPASRI2013_v2.0' 
NAME_MAVROULI = 'Mavrouli_et_al_2014'
NAME_TORRES = 'Torres_Corredor_et_al_2017'
# not all schemas are supported for all # data at the moment
# for full support all schemas should be included
# in all kind of files
# TODO: All should be fully supported!!!!
SCHEMAS_WITH_EXPOSURE = [NAME_SARA]
SCHEMAS_WITH_FRAGILITY = [NAME_SARA, NAME_HAZUS, NAME_SUPPASRI]
SCHEMAS_WITH_MAPPING = [NAME_SARA, NAME_SUPPASRI]
SCHEMAS_WITH_REPLACEMENT_COSTS = [NAME_SARA, NAME_SUPPASRI]

def nearly1(x):
    return 0.99 < x < 1.01

SourceTargetCombination = collections.namedtuple(
    'SourceTargetCombination',
    'source_schema target_schema'
)

def read_json(filename):
    with open(filename, 'rt') as input_file:
        data = json.load(input_file)
    return data

class TestTaxMappingFiles(unittest.TestCase):
    """
    This Test case is only for the taxonomy mapping files.
    """

    def setUp(self):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_files = glob.glob(os.path.join(current_dir, 'tax_schemamappings', '*.json'))

        self.json_data = []

        for json_file in json_files:
            with open(json_file, 'rt') as input_file:
                data = json.load(input_file)
                self.json_data.append(data)


    def test_all_tax_mapping_files_sum_to_1(self):
        """
        The sum for a source taxonomies over all
        target taxonomies must be 1.
        """

        for dataset in self.json_data:
            conv_matrix = dataset['conv_matrix']

            for source_taxonomy in conv_matrix.keys():
                conv_row = conv_matrix[source_taxonomy]

                sum_over_row = sum(conv_row.values())

                self.assertTrue(nearly1(sum_over_row))

    def test_tax_mapping_supported(self):
        """
        Test that all the taxomomy mapping files have the supported
        schemas.
        """


        schema_mappings = set()
        for dataset in self.json_data:
            source_schema = dataset['source_schema']
            target_schema = dataset['target_schema']
            schema_mappings.add(
                SourceTargetCombination(
                    source_schema=source_schema,
                    target_schema=target_schema,
                )
            )

        # we need to support the mapping from
        # sara to supparsi
        self.assertIn(
            SourceTargetCombination(
                source_schema=NAME_SARA, 
                target_schema=NAME_SUPPASRI,
            ),
            schema_mappings
        )

    def test_covers_full_sara_to_suppasri_for_all_tax(self):
        """
        Reads the taxomomy from the exposure json file for sara
        and checks if the all the taxonomies of sara are
        supported.
        """

        current_dir = os.path.dirname(os.path.abspath(__file__))
        exposure_file_sara = os.path.join(
            current_dir, 
            'assetmaster',
            'SARA_v1.0_asset_master.json',
        )

        exposure_sara = read_json(exposure_file_sara)

        sara_taxonomies = exposure_sara['taxonomies']

        covered_taxonomies = set()

        for dataset in self.json_data:
            source_schema = dataset['source_schema']
            target_schema = dataset['target_schema']

            if source_schema == NAME_SARA and target_schema == NAME_SUPPASRI:
                conv_matrix = dataset['conv_matrix']
                taxonomies = conv_matrix.keys()

                for tax in taxonomies:
                    covered_taxonomies.add(tax)

        not_covered_taxonomies = set(sara_taxonomies) - covered_taxonomies

        self.assertEqual(not_covered_taxonomies, set())



class TestDsMappingFiles(unittest.TestCase):
    """
    This test case is only for the damage state mapping
    files.
    """

    def setUp(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_files = glob.glob(os.path.join(current_dir, 'ds_schemamappings', '*', '*.json'))

        self.json_data = []
        for json_file in json_files:
            with open(json_file, 'rt') as input_file:
                data = json.load(input_file)
                self.json_data.append(data)


    def test_all_ds_mapping_files_sum_to_1(self):
        """
        The sum for each damage state mapping between
        schemas must be 1.
        """

        for dataset in self.json_data:
            # because the ds files have the target damage states
            # as toplevel keys, we can use the pandas dataframe
            conv_matrix = pd.DataFrame(dataset['conv_matrix'])

            self.assertTrue(all(conv_matrix.apply(sum, axis=1).apply(nearly1)))

    def test_tax_mapping_supported(self):
        """
        Test that all the taxomomy mapping files have the supported
        schemas.
        """


        schema_mappings = set()
        for dataset in self.json_data:
            source_schema = dataset['source_schema']
            target_schema = dataset['target_schema']
            schema_mappings.add(
                SourceTargetCombination(
                    source_schema=source_schema,
                    target_schema=target_schema,
                )
            )

        # we need to support the mapping from
        # sara to supparsi
        self.assertIn(
            SourceTargetCombination(
                source_schema=NAME_SARA, 
                target_schema=NAME_SUPPASRI,
            ),
            schema_mappings
        )

    def test_covers_full_sara_to_suppasri_for_all_tax(self):
        """
        Reads the taxonomy from the exposure json file for sara
        and checks if all the taxonomies of sara are supported.
        """


        current_dir = os.path.dirname(os.path.abspath(__file__))
        exposure_file_sara = os.path.join(
            current_dir, 
            'assetmaster',
            'SARA_v1.0_asset_master.json',
        )

        exposure_sara = read_json(exposure_file_sara)

        sara_taxonomies = exposure_sara['taxonomies']

        covered_taxonomies = set()

        for dataset in self.json_data:
            source_schema = dataset['source_schema']
            target_schema = dataset['target_schema']

            if source_schema == NAME_SARA and target_schema == NAME_SUPPASRI:
                source_taxonomy = dataset['source_taxonomy']

                covered_taxonomies.add(source_taxonomy)

        not_covered_taxonomies = set(sara_taxonomies) - covered_taxonomies

        self.assertEqual(not_covered_taxonomies, set())

    def test_covers_full_sara_to_suppasri_for_all_ds(self):
        """
        Test that we cover all source damage states
        in all the files for the sara to suppasri
        mappings.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fragility_file_sara = os.path.join(
            current_dir, 
            'modelprop',
            'SARA_v1.0_struct.json',
        )

        fragility_sara = read_json(fragility_file_sara)

        sara_ds = set(['0'] + [x[1:] for x in fragility_sara['meta']['limit_states']])

        fragility_file_suppasri = os.path.join(
            current_dir,
            'modelprop',
            'SUPPASRI2013_v2.0_struct.json'
        )

        fragility_suppasri = read_json(fragility_file_suppasri)

        suppasri_ds = set(['0'] + [x[1:] for x in fragility_suppasri['meta']['limit_states']])

        for dataset in self.json_data:
            source_taxonomy = dataset['source_taxonomy']
            target_taxonomy = dataset['target_taxonomy']

            if source_taxonomy == NAME_SARA and target_taxonomy == NAME_SUPPASRI:
                conv_matrix = pd.DataFrame(dataset['conv_matrix'])

                source_ds = set(conv_matrix.index)
                target_ds = set(conv_matrix.columns)

                # we need to support the full range of input damage states
                self.assertEqual(source_ds, sara_ds)

                for single_target_ds in target_ds:
                    # and the target damage states must be those
                    # that are defined
                    self.assertIn(single_target_ds, suppasri_ds)

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
        self.exposure_geo_data = ExposureGeoData.read_from_gpgk_files()

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
                'SA(1.0)': 'g',
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

    def test_all_taxonomies_from_gpgk_are_in_json_exposure_data(self):
        '''
        Test that all the taxonomies in the gpgk exposure files
        are included in the json files for the exposure.
        '''
        for schema in self.exposure_geo_data.get_schemas():
            tax_json = set(self.exposure_data.get_taxonomies(schema))
            for taxonomy in self.exposure_geo_data.get_taxonomies(schema):
                self.assertIn(taxonomy, tax_json)

    def test_damage_states_from_fragility_files_and_schema_mapping(self):
        '''
        Test that all damage states, so that we can be sure that all
        damage states from the fragility models are in the source
        for the schema mapping (so that we don't get an unexpected
        damage state) and that only those damage states as targets
        are used, that are also in the fragility models.
        '''

        source_schemas = self.schemamapping_data.get_source_schemas()
        fragility_schemas = self.fragility_data.get_schemas()

        for source_schema in source_schemas:
            self.assertIn(source_schema, fragility_schemas)

            source_taxonomies = self.schemamapping_data.get_source_taxonomies(source_schema)
            fragility_taxonomies = self.fragility_data.get_taxonomies_in_data(source_schema)
            fragility_damage_states = self.fragility_data.get_damage_states_in_meta_with_D(source_schema)

            for source_taxonomy in source_taxonomies:
                self.assertIn(source_taxonomy, fragility_taxonomies)

                source_damage_states = self.schemamapping_data.get_source_damage_states_from_conv_matrix_with_d(
                    source_schema,
                    source_taxonomy
                )

                # we only use defined damage states
                # as source
                for source_damage_state in source_damage_states:
                    if source_damage_state == 'D0':
                        continue
                    self.assertIn(source_damage_state, fragility_damage_states)

                # and we cover all defined damage states
                # as sources
                for fragility_damage_state in fragility_damage_states:
                    self.assertIn(fragility_damage_state, source_damage_state)

                source_damage_states_not_from_conv_matrix = self.schemamapping_data.get_source_damage_states_from_data_with_d(
                    source_schema,
                    source_taxonomy
                )

                # check that the source damage states from the conv_matrix and those
                # in the data section itself match
                for source_damage_state in source_damage_states:
                    self.assertIn(source_damage_state, source_damage_states_not_from_conv_matrix)
                for source_damage_states in source_damage_states_not_from_conv_matrix:
                    self.assertIn(source_damage_state, source_damage_states)


        target_schemas = self.schemamapping_data.get_target_schemas()

        for target_schema in target_schemas:
            self.assertIn(target_schema, fragility_schemas)

            target_taxonomies = self.schemamapping_data.get_taxonomies(target_schema)

            fragility_taxonomies = self.fragility_data.get_taxonomies_in_data(source_schema)
            fragility_damage_states = self.fragility_data.get_damage_states_in_meta_with_D(source_schema)

            for target_taxonomy in target_taxonomies:
                self.assertIn(target_taxonomy, fragility_taxonomies)

                target_damage_states = self.schemamapping_data.get_target_damage_states_from_conv_matrix_with_d(
                    target_schema,
                    target_taxonomy,
                )

                # here we only are interested, that all target damage states
                # are defined ones
                # we don't need to cover all defined as targets, as those may not be meaningful
                # and can be supposed to be zero
                for target_damage_state in target_damage_states:
                    if target_damage_state == 'D0':
                        continue
                    self.assertIn(target_damage_state, fragility_damage_states)

                target_damage_states_not_from_conv_matrix = self.schemamapping_data.get_target_damage_states_from_data_with_d(
                    target_schema,
                    target_taxonomy
                )

                for target_damage_state in target_damage_states:
                    self.assertIn(target_damage_state, target_damage_states_not_from_conv_matrix)

    def test_damage_states_for_replacement_costs(self):
        """
        Here we test that all replacement costs that are mentioned in
        the fragility model are in the replacement costs (and that all
        intermediate ones too).
        """

        repl_cost_schemas = self.replacementcost_data.get_schemas()
        fragility_schemas = self.fragility_data.get_schemas()

        for schema in repl_cost_schemas:
            self.assertIn(schema, fragility_schemas)

            repl_cost_taxonomies = self.replacementcost_data.get_taxonomies(schema)
            fragility_taxonomies = self.fragility_data.get_taxonomies_in_meta(schema)

            fragility_damage_states = self.fragility_data.get_damage_states_in_meta_with_D(schema)
            # damage states are like D1, D2, D3, etc.; but there is no D0
            # and we want them to be just numbers
            pure_number_damage_states = sorted([0] + [int(ds[1:]) for ds in fragility_damage_states])

            def all_but_last(damage_states):
                # as they are sorted
                return damage_states[:-1]

            def all_higher(all_damage_states, filter_damage_state):
                return [ds for ds in all_damage_states if ds > filter_damage_state]



            repl_cost_data = self.replacementcost_data.data[schema]['data']

            for repl_cost_dataset in repl_cost_data:
                taxonomy = repl_cost_dataset['taxonomy']
                loss_matrix = repl_cost_dataset['loss_matrix']

                # loss matrix contains a structure like this
                # {
                #   "0": {
                #     "1": 10,
                #     "2": 100,
                #     "3": 1000,
                #   },
                #   "1": {
                #     "2": 90,
                #     "3": 900,
                # ...
                # so that the keys are all *but the last* damage states
                # while the inner keys are all higher damage states 
                for ds in all_but_last(pure_number_damage_states):
                    self.assertIn(str(ds), loss_matrix.keys())

                    all_higher_ds = all_higher(pure_number_damage_states, ds)

                    for higher_ds in all_higher_ds:
                        self.assertIn(str(higher_ds), loss_matrix[str(ds)].keys())

        for schema in fragility_schemas:
            self.assertIn(schema, repl_cost_schemas)


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

    def get_damage_states_in_meta_with_D(self, schema):
        """
        Returns the damage states with D, so D0, D1, D2, ...
        """
        return self.data[schema]['meta']['limit_states']


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

class ExposureGeoData():
    def __init__(self, data):
        self.data = data

    @classmethod
    def read_from_gpgk_files(cls):
        taxonomies_by_schema = {}

        current_dir = os.path.dirname(os.path.abspath(__file__))
        assetmaster_dir = os.path.join(current_dir, 'assetmaster')
        gpkg_files = glob.glob(os.path.join(assetmaster_dir, '*.gpkg'))

        for gpkg_file in gpkg_files:
            taxonomies = set()
            geo_data = gpd.read_file(gpkg_file)
            for expo_str in geo_data['expo']:
                expo = pd.DataFrame(json.loads(expo_str))
                for tax in expo['Taxonomy']:
                    taxonomies.add(tax)

            base_filename = os.path.basename(gpkg_file)
            schema = base_filename.replace('_data.gpkg', '')
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

    def get_source_damage_states_from_conv_matrix_with_d(self, schema, taxonomy):
        result = set()
        for dataset in self.data:
            source_schema = dataset['source_schema']
            source_taxonomy = dataset['source_taxonomy']
            if source_schema == schema and source_taxonomy == taxonomy:
                conv_matrix = dataset['conv_matrix']
                result = conv_matrix.keys()

        result = ['D' + ds for ds in result]
        return result

    def get_source_damage_states_from_data_with_d(self, schema, taxonomy):
        result = set()
        for dataset in self.data:
            source_schema = dataset['source_schema']
            source_taxonomy = dataset['source_taxonomy']
            if source_schema == schema and source_taxonomy == taxonomy:
                result = result['source_damage_states']

        result = ['D' + ds for ds in result]
        return result

    def get_target_damage_states_from_conv_matrix_with_d(self, schema, taxonomy):
        result = set()
        for dataset in self.data:
            target_schema = dataset['target_schema']
            target_taxonomy = dataset['target_taxonomy']
            if target_schema == schema and target_taxonomy == taxonomy:
                conv_matrix = dataset['conv_matrix']
                for source_damage_state in conv_matrix.keys():
                    for target_damage_state in source_damage_state.keys():
                        result.add(target_damage_state)
        result = ['D' + ds for ds in result]
        return result

    def get_target_damage_states_from_data_with_d(self, schema, taxonomy):
        result = set()
        for dataset in self.data:
            target_schema = dataset['target_schema']
            target_taxonomy = dataset['target_taxonomy']
            if target_schema == schema and target_taxonomy == taxonomy:
                result = dataset['target_damage_states']
        result = ['D' + ds for ds in result]
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
