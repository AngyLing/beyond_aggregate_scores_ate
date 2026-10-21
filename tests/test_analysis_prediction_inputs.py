"""Fresh-analysis input and comparison contracts on artificial text only."""
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ate_paper.analysis import io
from ate_paper.predictions import CorpusIndex, release_record, sha256

class AnalysisPredictionInputsTest(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name).resolve()
        self.row={'dataset':'ACL RD-TEC 2.0','doc_id':'toy','sent_id':'1','split':'test',
                  'version':'toy','text':'Alpha beta','unit_type':'span_mention',
                  'start':0,'end':5,'surface':'Alpha','norm':'alpha','class_label':'Term'}
        self.index=CorpusIndex.build([self.row])
        self.entry={'dataset':self.row['dataset'],'group':'supervised','run':'toy_seed42',
                    'release_path':'predictions/toy.jsonl.gz'}
        self.fresh=self.root/'new/acl/supervised/toy_seed42/predictions.jsonl'
        self.fresh.parent.mkdir(parents=True)
    def write(self,row):self.fresh.write_text(json.dumps(row)+'\n')
    def test_fresh_raw_rows_use_text_identity_not_old_measurements(self):
        self.write(self.row)
        rows,path=io.load_prediction_rows(self.root,self.entry,self.index,self.root/'new')
        self.assertEqual(rows[0]['text'],'Alpha beta')
        self.assertEqual(rows[0]['surface'],'Alpha')
        self.assertEqual(path,self.fresh)
    def test_changed_text_or_surface_is_rejected(self):
        for changed in [{**self.row,'text':'Other text'},{**self.row,'surface':'wrong'}]:
            self.write(changed)
            with self.assertRaises(ValueError):io.load_prediction_rows(self.root,self.entry,self.index,self.root/'new')
    def test_missing_is_error_but_existing_empty_file_means_zero_predictions(self):
        with self.assertRaises(FileNotFoundError):io.load_prediction_rows(self.root,self.entry,self.index,self.root/'new')
        self.fresh.write_text('')
        self.assertEqual(io.load_prediction_rows(self.root,self.entry,self.index,self.root/'new')[0],[])
    def test_public_mode_preserves_manifest_checksum(self):
        path=self.root/self.entry['release_path'];path.parent.mkdir()
        with gzip.open(path,'wt') as stream:stream.write(json.dumps(release_record(self.row))+'\n')
        entry={**self.entry,'release_sha256':sha256(path)}
        self.assertEqual(len(io.load_prediction_rows(self.root,entry,self.index)[0]),1)
        entry['release_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA-256'):io.load_prediction_rows(self.root,entry,self.index)
    def test_full_test_keeps_empty_units_and_rejects_truncation(self):
        empty={**self.row,'doc_id':'empty','start':None,'end':None,'surface':None,'norm':None,'class_label':None}
        with patch.dict(io.TEST_COUNTS,{self.row['dataset']:{'documents':2,'units':2,'spans':1}}):
            io.validate_test_rows([self.row,empty],self.row['dataset'])
            with self.assertRaisesRegex(ValueError,'incomplete'):io.validate_test_rows([self.row],self.row['dataset'])
    def test_new_values_are_reported_as_differences(self):
        result=io.compare_rows([{'run':'toy','f1':0.8}],[{'run':'toy','f1':'0.7'}],('run',))
        self.assertFalse(result['all_saved_values_match'])
        self.assertAlmostEqual(result['differences'][0]['delta'],0.1)
    def test_missing_reference_rows_and_undefined_values_are_visible(self):
        result=io.compare_rows([{'run':'a','metric':float('nan')}],[{'run':'a','metric':'nan'},{'run':'b','metric':'1'}],('run',))
        self.assertEqual(result['difference_count'],1)
        self.assertEqual(result['differences'][0]['field'],'__row__')
    def test_equivalent_undefined_encodings_do_not_report_numeric_differences(self):
        for actual in (None, '', float('nan')):
            for saved in (None, '', 'nan', 'NaN'):
                with self.subTest(actual=actual, saved=saved):
                    result=io.compare_rows([{'run':'a','metric':actual}],
                                           [{'run':'a','metric':saved}],('run',))
                    self.assertTrue(result['all_saved_values_match'])
        self.assertFalse(io.compare_rows([{'run':'a','metric':None}],
                                         [{'run':'a','metric':'0'}],('run',))['all_saved_values_match'])
    def test_saved_supervised_loader_rejects_missing_term_free_units(self):
        dataset=self.row['dataset']
        source=self.root/'results/v1_profile_verified_document_analysis_run_scores.csv'
        source.parent.mkdir()
        with source.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['dataset','group','run'])
            writer.writeheader()
            writer.writerows({'dataset':dataset,'group':'supervised','run':f'toy{i}'} for i in range(50))
        gold=self.root/'data/acl/test.jsonl';gold.parent.mkdir(parents=True)
        gold.write_text(json.dumps(self.row)+'\n')
        with patch.dict(io.TEST_COUNTS,{dataset:{'documents':2,'units':2,'spans':1}}):
            with self.assertRaisesRegex(ValueError,'incomplete'):
                io.load_fixed_dataset(self.root,self.root/'data',dataset)

if __name__=='__main__':unittest.main()
