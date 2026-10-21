"""Artificial-only contracts for full panel counts, undefined ratios and shared draws."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError:
    panels = None
else:
    from ate_paper.analysis import panels


def row(doc,start,end,label='Specific',dataset='ACL RD-TEC 2.0',split='test'):
    text='alpha beta gamma'
    return {'dataset':dataset,'split':split,'doc_id':doc,'sent_id':'1','text':text,
            'unit_type':'span_mention','start':start,'end':end,
            'surface':text[start:end] if start is not None else None,
            'norm':text[start:end].lower() if start is not None else None,'class_label':label}


@unittest.skipIf(panels is None, "Install the analysis profile from requirements.lock for numerical tests")
class PanelsContracts(unittest.TestCase):
    def test_data_stats_preserve_duplicate_class_offsets_and_negative_units(self):
        rows=[row('a',0,16),row('a',0,10),row('a',0,5),row('a',0,5,'Common'),
              row('a',0,5),row('empty',None,None,None)]
        summary,classes=panels.data_statistics('ACL RD-TEC 2.0','test',rows)
        self.assertEqual((summary['documents'],summary['text_units'],summary['unique_spans'],summary['mention_class_pairs']),(2,2,3,4))
        self.assertEqual((summary['inner'],summary['outer'],summary['both'],summary['nonflat']),(2,2,1,3))
        self.assertEqual(summary['term_free_units'],1)
        self.assertEqual(summary['exact_duplicate_annotation_rows'],1)
        self.assertEqual(next(r for r in classes if r['class_label']=='Specific')['annotation_rows'],4)

    def test_typed_pairs_and_untyped_na_are_separate_from_offsets(self):
        gold=[row('a',0,10),row('a',0,5),row('a',0,5,'Common'),row('empty',None,None,None)]
        pred=[row('a',0,5),row('a',0,5)]
        specs=[{'run':'typed','model':'typed','group':'llm','class_capable':'True'},
               {'run':'untyped','model':'xlmr_span_unlabeled','group':'supervised','typed_applicable':'False'}]
        result=panels.run_diagnostics('ACL RD-TEC 2.0',specs,gold,{'typed':pred,'untyped':pred},gold)
        scores=result[0]
        self.assertEqual(scores[0]['gold'],2)
        self.assertEqual(scores[0]['typed_gold'],3)
        self.assertEqual(scores[0]['predicted'],1)
        self.assertAlmostEqual(scores[0]['f1'],2/3)
        self.assertAlmostEqual(scores[0]['typed_f1'],0.5)
        self.assertIsNone(scores[1]['typed_f1'])
        self.assertIsNone(scores[1]['ClassGap'])
        self.assertNotIn('gap:ClassGap',result[1]['untyped'])

    def test_sample_sd_and_single_output_na(self):
        fields=('precision','recall','f1','typed_precision','typed_recall','typed_f1','soft_precision','soft_recall','soft_f1','inner_recall','outer_recall','ClassGap','BoundaryGap','NestedGap')
        runs=[{'run':'a'+str(i),'model':'a','group':'supervised','typed_applicable':True,**{k:value for k in fields}} for i,value in enumerate([0,0,0,0,1])]
        runs+=[{'run':'b','model':'b','group':'llm','typed_applicable':True,**{k:0.2 for k in fields}}]
        summary=panels.configuration_scores('fake',runs)
        self.assertAlmostEqual(summary[0]['f1_mean'],0.2)
        self.assertAlmostEqual(summary[0]['f1_sample_sd'],math.sqrt(0.2))
        self.assertIsNone(summary[1]['f1_sample_sd'])

    def test_many_to_one_witnesses_are_not_a_matching(self):
        keys=[('a','1',0,5),('a','1',1,6)]
        witness={('a','1',0,6)}
        self.assertEqual(panels.near_boundary_mask(keys,witness).tolist(),[True,True])

    def test_shared_document_bootstrap_invalidates_incomplete_constituent_means(self):
        gold={('d1','1',0,5),('d2','1',0,5)}
        predictions=[{('d1','1',0,5),('d1','1',10,14)},
                     {('d2','1',0,5),('d2','1',10,14)},
                     {('d1','1',0,5),('d2','1',0,5),('d1','1',10,15)},set()]
        runs=[{'run':'a0','model':'a','group':'supervised','precision':0.5,'recall':0.5},
              {'run':'a1','model':'a','group':'supervised','precision':0.5,'recall':0.5},
              {'run':'b','model':'b','group':'llm','precision':2/3,'recall':1},
              {'run':'c','model':'c','group':'gazetteer','precision':0,'recall':0}]
        arrays,pairs,groups,within,comparisons,doc_counts,flat=panels.prepare_output_sets('toy',runs,predictions,gold,['d2','d1'])
        self.assertEqual(arrays['doc_ids'].tolist(),['d2','d1'])
        ab=[r for r in pairs if r['run_a']=='a0' and r['run_b']=='b'][0]
        self.assertEqual(ab['fp_turnover'],1)
        self.assertEqual(ab['fp_no_witness_share'],0)
        weights=np.array([[1,1],[2,0],[0,2]])
        boot=panels.bootstrap_summaries('toy',weights,arrays,pairs,comparisons)
        ab_tp=next(r for r in boot if r['model_a']=='a' and r['model_b']=='b' and r['target']=='tp_turnover')
        self.assertEqual((ab_tp['between_valid'],ab_tp['within_reference_valid'],ab_tp['excess_valid']),(1,1,1))
        self.assertEqual((ab_tp['between_point'],ab_tp['within_reference_point'],ab_tp['excess_point']),(0,1,-1))
        bc=next(r for r in boot if r['model_a']=='b' and r['model_b']=='c' and r['target']=='tp_turnover')
        self.assertIsNone(bc['within_reference_point'])
        self.assertEqual(bc['excess_valid'],0)

    def test_perfect_outputs_allow_empty_fp_union(self):
        gold={('d','1',0,5)}
        runs=[{'run':'a','model':'a'},{'run':'b','model':'b'}]
        arrays,pairs,*_=panels.prepare_output_sets('toy',runs,[gold,gold],gold,['d','negative'])
        self.assertEqual(arrays['run_fp'].shape,(2,2))
        self.assertEqual(arrays['run_fp'].sum(),0)
        self.assertTrue(math.isnan(pairs[0]['fp_turnover']))
        self.assertTrue(math.isnan(pairs[0]['fp_no_witness_share']))

    def test_reference_differences_are_separate_and_do_not_fail_new_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);result_dir=root/'results';result_dir.mkdir()
            score_path=result_dir/'v1_profile_verified_document_analysis_run_scores.csv'
            score_path.write_text('dataset,run,f1\nACL RD-TEC 2.0,new-run,0.7\n')
            outputs={name:[] for name in ('run_scores','run_pairs','configuration_pairs','bootstrap','profiles','gap_context')}
            outputs['run_scores']=[{'dataset':'ACL RD-TEC 2.0','run':'new-run','f1':0.5,'new_field':99}]
            report,paths=panels.compare_saved_outputs(root,outputs,skip_bootstrap=True)
            comparison=next(r for r in report['comparisons'] if r['dataset']=='ACL RD-TEC 2.0' and r['output']=='run_scores')
            self.assertEqual(comparison['status'],'DIFFERENCES_REPORTED')
            self.assertEqual(comparison['difference_count'],1)
            self.assertAlmostEqual(comparison['differences'][0]['delta'],-0.2)
            self.assertEqual(outputs['run_scores'][0]['f1'],0.5)
            self.assertEqual(paths,{score_path})
            self.assertGreater(report['references_missing'],0)

    def test_mock_full_roster_point_workflow_never_reuses_saved_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'fake';out=root/'build/panels'
            datasets=panels.io.DATASETS
            payload={}
            for dataset in datasets:
                ds=panels.io.SLUGS[dataset]
                for split in ('train','dev','test'):
                    rows=[row(split+'-positive',0,5,dataset=dataset,split=split),
                          row(split+'-positive',0,10,dataset=dataset,split=split),
                          row(split+'-empty',None,None,None,dataset=dataset,split=split)]
                    p=source/ds/(split+'.jsonl');p.parent.mkdir(parents=True,exist_ok=True)
                    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    if split=='test':test=rows
                n_supervised=11 if ds=='cl' else 10
                specs=[]
                for i in range(n_supervised):
                    for seed in range(42,47):specs.append({'dataset':dataset,'group':'supervised','model':f'model{i:02d}','run':f'model{i:02d}_seed{seed}','seed':str(seed),'typed_applicable':'True'})
                for i in range(8):specs.append({'dataset':dataset,'group':'llm','model':f'llm{i}','run':f'llm{i}','typed_applicable':'True'})
                for i in range(2):specs.append({'dataset':dataset,'group':'gazetteer','model':f'gaz{i}','run':f'gaz{i}','typed_applicable':'True'})
                # Deliberately unequal predictions while specs carry no scores.
                predictions={s['run']:[test[0]] for s in specs}
                payload[dataset]=(specs,test,predictions,[source/ds/'test.jsonl'])
            with mock.patch.object(panels.io,'load_analysis_rows',side_effect=lambda _root,_data,dataset,**kw:payload[dataset],create=True),mock.patch.object(panels.io,'load_document_draws',side_effect=AssertionError('draws must not load under --skip-bootstrap'),create=True):
                manifest=panels.reproduce_panels(root,source,out,prediction_root=root/'fresh',skip_bootstrap=True)
            self.assertEqual(sum(s['runs'] for s in manifest['scope']),185)
            self.assertEqual(sum(s['configuration_pairs'] for s in manifest['scope']),590)
            self.assertEqual(manifest['protocol']['prediction_mode'],'fresh')
            text=(out/'run_scores.csv').read_text()
            self.assertIn('0.6666666666666666',text)
            self.assertFalse((out/'bootstrap.csv').exists())
            self.assertTrue(manifest['inputs_unchanged'])
            tracked = {Path(item['path']).resolve() for item in manifest['inputs']}
            self.assertIn(Path(panels.__file__).resolve().parents[1] / 'io.py', tracked)
            self.assertIn(Path(panels.__file__).resolve().parent / 'matching.py', tracked)
            with self.assertRaises(FileExistsError):panels.reproduce_panels(root,source,out,skip_bootstrap=True)
            def fake_draws(_root,dataset,unit_docs):
                docs=sorted(unit_docs,reverse=True)
                weights=np.array([[1,1],[2,0],[0,2]])
                draw_path=source/panels.io.SLUGS[dataset]/'toy_draws.npz'
                np.savez_compressed(draw_path,doc_ids=np.array(docs),weights=weights)
                return docs,weights,draw_path
            with mock.patch.object(panels.io,'load_analysis_rows',side_effect=lambda _root,_data,dataset,**kw:payload[dataset],create=True),mock.patch.object(panels.io,'load_document_draws',side_effect=fake_draws,create=True):
                full=panels.reproduce_panels(root,source,root/'build/with_bootstrap',prediction_root=root/'fresh')
            self.assertEqual((root/'build/with_bootstrap/bootstrap.csv').read_text().count('\n'),2361)
            self.assertEqual(sum(s['draws'] for s in full['scope']),9)
            self.assertIn('A08',full['protocol']['analysis_blocks'])
            self.assertNotIn('A08',manifest['protocol']['analysis_blocks'])
            with np.load(root/'build/with_bootstrap/acl_document_arrays.npz',allow_pickle=False) as arrays:
                self.assertEqual(arrays['weights'].tolist(),[[1,1],[2,0],[0,2]])


if __name__=='__main__':
    unittest.main(verbosity=2)
