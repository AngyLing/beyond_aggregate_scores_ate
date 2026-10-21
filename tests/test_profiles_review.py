"""Only artificial fixtures; no corpus, saved prediction or draw-array reads."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ate_paper.analysis import io, profiles as fixed
DS = 'ACL RD-TEC 2.0'


def golden_row(split='test'):
    return dict(dataset=DS,split=split,doc_id='positive',sent_id='1',text='alpha beta',
                start=0,end=5,surface='alpha',norm='alpha',class_label='Specific',unit_type='span_mention')


class ProfilesReviewContracts(unittest.TestCase):
    def fake_workflow(self, module, *, omit_selected=False):
        """Mock scoring, not comparison: actual compare_rows sees extra reference runs."""
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);root=base/'repo';gold=base/'local';fresh=base/'fresh'
            root.mkdir();gold.mkdir();fresh.mkdir()
            out=root/'build/profiles'
            score=dict(dataset=DS,model='m',group='supervised',seed=42,run='m_seed42',precision=0.5,recall=0.5,f1=0.5,class_capable=True)
            other=dict(score,run='m_seed43',seed=43)
            computed=[score,other]
            reference=[score] if omit_selected else list(computed)
            reference += [dict(score,group='llm',model='llm',run='llm'),dict(score,group='gazetteer',model='gaz',run='gaz')]
            profile=dict(dataset=DS,run='m_seed42',family='length',slice='1',support=30,active=True,value=0.5)
            summary={'dataset':DS}
            grid=[{'dataset':DS,'pr_margin':pm,'profile_margin':fm,**module.screens([],[],[],pm,fm)}
                  for pm in (.02,.03,.05,.10) for fm in (.05,.10,.15,.20,.30)]
            paths={
                'manifest':root/'predictions/manifest.csv',
                'scores':root/'results/acl_v1_profile_recheck_run_scores.csv',
                'profiles':root/'results/acl_v1_profile_recheck_profiles.csv',
                'pairs':root/'results/acl_v1_profile_recheck_configuration_pairs.csv',
                'summary':root/'results/profile_closeness_calibration_2026_10_13/calibration_summary.csv',
                'grid':root/'results/profile_closeness_calibration_2026_10_13/threshold_grid.csv',
                'script':root/'scripts/reproduce_profiles.py',
            }
            pred_paths=[fresh/'acl/supervised'/name/'predictions.jsonl' for name in ('m_seed42','m_seed43')]
            for path in [*paths.values(),*pred_paths]:
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic\n')
            rows={paths['manifest']:[],paths['scores']:reference,paths['profiles']:[profile],paths['pairs']:[],paths['summary']:[summary],paths['grid']:grid}
            with mock.patch.object(module,'DATASETS',{DS:'acl'}), \
                 mock.patch.object(module,'read_csv',side_effect=lambda p:rows[p]), \
                 mock.patch.object(module,'rebuild_dataset',return_value=(computed,{}, {},[profile],[],[],pred_paths)), \
                 mock.patch.object(module,'calibrate',return_value=([],[],[],summary)):
                result=module.reproduce_profiles(root,gold,out,prediction_root=fresh)
            return result

    def test_fresh_comparison_ignores_other_panel_groups(self):
        after=self.fake_workflow(fixed)
        self.assertEqual(after['checks'][DS]['scores']['difference_count'],0)
        self.assertEqual(after['checks'][DS]['scores']['computed_rows'],2)
        self.assertEqual(after['checks'][DS]['scores']['reference_rows'],2)

    def test_fresh_comparison_still_reports_missing_selected_run(self):
        result=self.fake_workflow(fixed,omit_selected=True)
        self.assertEqual(result['checks'][DS]['scores']['difference_count'],1)
        self.assertEqual(result['checks'][DS]['scores']['differences'][0]['field'],'__row__')

    def test_external_prediction_receipts_preserve_distinct_relative_paths(self):
        result=self.fake_workflow(fixed)
        names=[r['path'] for r in result['inputs'] if r['path'].startswith('LOCAL_PREDICTIONS/')]
        self.assertEqual(names,['LOCAL_PREDICTIONS/acl/supervised/m_seed42/predictions.jsonl',
                                'LOCAL_PREDICTIONS/acl/supervised/m_seed43/predictions.jsonl'])
        self.assertEqual(len(set(names)),2)
        shared = Path(fixed.__file__).resolve().parents[1] / 'io.py'
        receipts = [item for item in result['implementation'] if item['path'] == shared.name]
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]['sha256'], fixed.sha256(shared))

    def test_saved_and_fresh_profiles_reject_missing_negative_test_unit(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);data=base/'processed';(data/'acl').mkdir(parents=True)
            for split in ('train','test'):
                (data/'acl'/f'{split}.jsonl').write_text(json.dumps(golden_row(split))+'\n')
            expected={DS:{'documents':2,'units':2,'spans':1}}
            for pred_root in (None,base/'fresh'):
                with self.subTest(mode=pred_root),mock.patch.object(io,'TEST_COUNTS',expected), \
                     mock.patch.object(io,'load_prediction_rows',side_effect=AssertionError('must reject before reading predictions')):
                    with self.assertRaisesRegex(ValueError,'Prepared test is incomplete'):
                        fixed.rebuild_dataset(base,data,DS,[],prediction_root=pred_root)


if __name__=='__main__':
    unittest.main(verbosity=2)
