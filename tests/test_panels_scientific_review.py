"""Small independent examples for A01–A08, never article inputs or saved arrays."""
import itertools
import math
import unittest

try:
    import numpy as np
except ImportError:
    panels = None
else:
    from ate_paper.analysis import panels, profiles


def row(doc,start,end,label='Specific'):
    text='abcdefghij'
    return dict(dataset='toy',split='test',doc_id=doc,sent_id='1',text=text,unit_type='span_mention',
                start=start,end=end,surface=text[start:end] if start is not None else None,
                norm=text[start:end] if start is not None else None,class_label=label)


def oracle_greedy(gold,pred):
    """Directly ordered edge list: highest IoU, then descending saved indices."""
    offsets=[]
    for rows in (gold,pred):
        unique=[]
        for r in rows:
            k=(r['doc_id'],r['sent_id'],r['start'],r['end'])
            if r['start'] is not None and k not in unique:unique.append(k)
        offsets.append(unique)
    candidates=[]
    for gi,g in enumerate(offsets[0]):
        for pi,p in enumerate(offsets[1]):
            if g[:2]!=p[:2]:continue
            overlap=max(0,min(g[3],p[3])-max(g[2],p[2]));union=max(g[3],p[3])-min(g[2],p[2])
            if overlap/union>=.5:candidates.append((overlap/union,gi,pi))
    used_g=set();used_p=set()
    for _,gi,pi in sorted(candidates,reverse=True):
        if gi not in used_g and pi not in used_p:used_g.add(gi);used_p.add(pi)
    return len(used_g)


def finite_ratio(a,b):
    return a/b if b else math.nan


def reference_draws(gold,predictions,docs,weights):
    """Set/dictionary loops; no production profiles, membership arrays or statistics."""
    flat={g for g in gold if not any(g!=h and g[:2]==h[:2] and
                                   ((g[2]<=h[2] and h[3]<=g[3]) or (h[2]<=g[2] and g[3]<=h[3])) for h in gold)}
    pairs=list(itertools.combinations(range(len(predictions)),2))
    answers={t:[] for t in panels.TARGETS}
    def witness(span,other):
        for k in other:
            if k[:2]!=span[:2]:continue
            overlap=max(0,min(k[3],span[3])-max(k[2],span[2]));union=max(k[3],span[3])-min(k[2],span[2])
            if overlap/union>=.5:return True
        return False
    for weight in weights:
        per_target={t:[] for t in answers}
        def count(spans):return sum(int(weight[docs.index(k[0])]) for k in spans)
        for i,j in pairs:
            a,b=predictions[i],predictions[j]
            for name,universe in [('tp',gold),('fp',None),('flat_tp',flat)]:
                x=a-gold if universe is None else a&universe
                y=b-gold if universe is None else b&universe
                denominator=min(count(x),count(y))
                per_target[name+'_turnover'].append(finite_ratio(denominator-count(x&y),denominator))
            x,y=a-gold,b-gold
            no_x={k for k in x if not witness(k,b)};no_y={k for k in y if not witness(k,a)}
            per_target['fp_no_witness_share'].append((finite_ratio(count(no_x),count(x))+finite_ratio(count(no_y),count(y)))/2)
        for name,values in per_target.items():answers[name].append(values)
    return {name:np.array(values) for name,values in answers.items()}


@unittest.skipIf(panels is None, "Install the analysis profile from requirements.lock for numerical tests")
class IndependentScientificContracts(unittest.TestCase):
    def test_boundary_gap_ties_follow_saved_order_and_unique_offsets(self):
        gold=[row('d',0,1),row('d',0,3),row('d',0,1,'Other')]
        pred=[row('d',0,2),row('d',1,3),row('d',0,2)]
        self.assertEqual(oracle_greedy(gold,pred),2)
        self.assertEqual(profiles.soft_matches(gold,pred),2)
        reversed_pred=list(reversed(pred[:2]))
        self.assertEqual(oracle_greedy(gold,reversed_pred),1)
        self.assertEqual(profiles.soft_matches(gold,reversed_pred),1)
        desc,_=profiles.describe_gold([],gold)
        self.assertEqual(profiles.gap_context(gold,pred,desc,True)['BoundaryGap'],1)
        self.assertEqual(profiles.gap_context(gold,reversed_pred,desc,True)['BoundaryGap'],.5)

    def test_typed_multi_class_offsets_and_empty_predictions_keep_applicability(self):
        gold=[row('d',0,1,'A'),row('d',0,1,'B'),row('d',0,1,'A'),row('negative',None,None,None)]
        specs=[dict(run='typed',model='typed',group='supervised',typed_applicable=True),
               dict(run='untyped',model='mbert_unlabeled_bio',group='supervised',typed_applicable=False),
               dict(run='empty',model='typed_empty',group='llm',typed_applicable=True)]
        pred={'typed':[row('d',0,1,'A')],'untyped':[row('d',0,1,None)],'empty':[]}
        scores,profiles_by_run,supports,exported,*_=panels.run_diagnostics('toy',specs,gold,pred,gold)
        typed,untyped,empty=scores
        self.assertEqual((typed['gold'],typed['typed_gold'],typed['typed_tp']),(1,2,1))
        self.assertAlmostEqual(typed['typed_precision'],1)
        self.assertAlmostEqual(typed['typed_recall'],.5)
        self.assertAlmostEqual(typed['typed_f1'],2/3)
        self.assertAlmostEqual(typed['ClassGap'],1/3)
        self.assertTrue(all(untyped[k] is None for k in ('typed_precision','typed_recall','typed_f1','ClassGap')))
        self.assertNotIn('gap:ClassGap',profiles_by_run['untyped'])
        self.assertEqual((empty['typed_precision'],empty['typed_recall'],empty['typed_f1'],empty['ClassGap']),(0,0,0,0))
        self.assertIn('gap:ClassGap',profiles_by_run['empty'])
        self.assertNotIn('gap:NestedGap',profiles_by_run['typed'])

    def test_full_profiles_keep_rare_and_empty_cells_and_class_overlap(self):
        gold=[]
        for length,count in ((1,29),(2,30)):
            text='x' if length==1 else 'x y'
            for i in range(count):
                r=row(f'l{length}-{i}',0,len(text),'A')
                r.update(text=text,surface=text,norm=text)
                gold.append(r)
        # A second class and an exact duplicate at the first offset do not inflate support.
        gold += [dict(gold[0],class_label='B'),dict(gold[0])]
        train=[dict(gold[0],doc_id='train1',split='train'),
               dict(gold[0],doc_id='train1',split='train',class_label='B'),
               dict(gold[0],doc_id='train2',split='train')]
        descriptors,slices=profiles.describe_gold(train,gold)
        self.assertEqual(descriptors[profiles.base_key(gold[0])]['train_count'],2)
        self.assertEqual(len(slices['train_frequency','2-5']),29)
        self.assertEqual(len(slices['gold_class','A']),59)
        self.assertEqual(len(slices['gold_class','B']),1)
        specs=[dict(run='m',model='m',group='llm',typed_applicable=True)]
        _,active,_,exported,*_=panels.run_diagnostics('toy',specs,gold,{'m':gold[:29]},train)
        cells={(r['family'],r['slice']):r for r in exported}
        self.assertEqual((cells['length','1']['support'],cells['length','1']['active'],cells['length','1']['value']),(29,False,1))
        self.assertEqual((cells['length','2']['support'],cells['length','2']['active'],cells['length','2']['value']),(30,True,0))
        self.assertEqual((cells['length','3']['support'],cells['length','3']['active'],cells['length','3']['value']),(0,False,None))
        self.assertNotIn('length:1',active['m'])
        self.assertIn('length:2',active['m'])
        summary,classes=panels.data_statistics('toy','test',gold+[row('negative',None,None,None)])
        self.assertEqual((summary['documents'],summary['text_units'],summary['unique_spans'],summary['mention_class_pairs'],summary['annotation_rows']),(60,60,59,60,61))
        self.assertEqual((summary['term_free_units'],summary['exact_duplicate_annotation_rows'],summary['flat']),(1,1,59))

    def test_profile_distance_nearest_rank_and_seed_screens_use_fixed_populations(self):
        supports={'length:1':30,'length:2':70,'gap:BoundaryGap':None}
        d=profiles.distance({'length:1':0.,'length:2':0.,'gap:BoundaryGap':0.},
                            {'length:1':.1,'length:2':.2,'gap:BoundaryGap':.15},supports)
        self.assertAlmostEqual(d['maximum'],.2)
        self.assertAlmostEqual(d['weighted_family_maximum'],.17)
        self.assertEqual(d['worst_coordinate'],'length:2')
        runs=[];values={}
        for model,offset in [('a',0),('b',.01)]:
            for seed in range(42,47):
                value=(seed-42)*.01+offset;name=f'{model}_seed{seed}'
                runs.append(dict(dataset='toy',model=model,run=name,seed=seed,precision=value,recall=value))
                values[name]={'length:1':value}
        within,cross,means,summary=profiles.calibrate('toy',runs,values,{'length:1':30})
        self.assertEqual((len(within),len(cross),len(means)),(20,25,1))
        self.assertAlmostEqual(means[0]['delta_p'],.01)
        self.assertAlmostEqual(means[0]['maximum'],.01)
        # Twenty within pairs: 8 x .01, 6 x .02, 4 x .03, 2 x .04.
        self.assertAlmostEqual(summary['pr_q80'],.03)
        self.assertAlmostEqual(summary['profile_q80'],.03)
        self.assertAlmostEqual(summary['joint_scale_q80'],1.5)
        self.assertAlmostEqual(profiles.nearest_rank([0.,0.,0.,1.,1.],.8),1.)
        self.assertEqual(summary['original_mean_configuration_pairs_pass'],1)
        self.assertEqual(summary['original_stable_configuration_pairs_pass'],0)
        self.assertEqual(summary['componentwise_q80_stable_configuration_pairs_pass'],1)
        self.assertEqual(summary['loo_radial_pass_total'],18)  # Both tied .03 pairs pass each holdout.
        # A mean pass alone is insufficient: require at least 20 of the 25 actual seed pairs.
        mean=[dict(model_a='a',model_b='b',pr_max=0.,maximum=0.)]
        for n,wanted in [(19,0),(20,1)]:
            artificial=[dict(model_a='a',model_b='b',pr_max=0. if i<n else 1.,maximum=0.) for i in range(25)]
            self.assertEqual(profiles.screens([],artificial,mean,.02,.05)['stable_configuration_pairs_pass'],wanted)

    def test_four_document_estimands_and_mixed_within_references_match_loop_oracle(self):
        docs=['negative','d2','d1']
        g1=('d1','1',0,4);g2=('d1','1',1,3);g3=('d2','1',0,2);gold={g1,g2,g3}
        predictions=[{g1,('d1','1',6,8),('negative','1',0,2)},
                     {g2,g3,('d1','1',7,9)},
                     {g1,g2,g3,('negative','1',1,3)},
                     {g1,g3,('d1','1',6,9)},set(),gold]
        runs=[dict(run=name,model=model,group='supervised' if model in ('a','b') else 'llm',precision=.5,recall=.5)
              for name,model in [('a42','a'),('a43','a'),('b42','b'),('b43','b'),('c','c'),('d','d')]]
        weights=np.array([[1,1,1],[3,0,0],[0,3,0],[0,0,3]])
        arrays,pairs,groups,within,comparisons,*_=panels.prepare_output_sets('toy',runs,predictions,gold,docs)
        reference=reference_draws(gold,predictions,docs,weights)
        for target in panels.TARGETS:
            with self.subTest(target=target):
                np.testing.assert_allclose(panels.bootstrap_values(weights,arrays,target),reference[target],rtol=0,atol=1e-12,equal_nan=True)
        summaries=panels.bootstrap_summaries('toy',weights,arrays,pairs,comparisons)
        pair_ids=list(itertools.combinations(range(len(runs)),2))
        def strict_mean(values,indices):
            return sum(values[i] for i in indices)/len(indices) if indices else math.nan
        for a,b,between,wa,wb in comparisons:
            for target in panels.TARGETS:
                rows=[]
                for v in reference[target]:
                    bt=strict_mean(v,between)
                    wr=[strict_mean(v,w) for w in (wa,wb) if w]
                    # A side with zero denominator invalidates the reference even if the other is finite.
                    ref=math.nan if not wr or any(math.isnan(x) for x in wr) else max(wr)
                    rows.append((bt,ref,bt-ref))
                computed=next(r for r in summaries if (r['model_a'],r['model_b'],r['target'])==(a,b,target))
                for field,column in [('between',0),('within_reference',1),('excess',2)]:
                    valid=[r[column] for r in rows if math.isfinite(r[column])]
                    self.assertEqual(computed[field+'_valid'],len(valid),(a,b,target,field))
                    if valid:
                        lo,hi=np.quantile(valid,[.025,.975])
                        self.assertAlmostEqual(computed[field+'_ci_low'],lo)
                        self.assertAlmostEqual(computed[field+'_ci_high'],hi)
                    else:self.assertIsNone(computed[field+'_ci_low'])
        cd=[r for r in summaries if (r['model_a'],r['model_b'])==('c','d')]
        self.assertTrue(all(r['within_sides']==0 and r['within_reference_valid']==0 and r['excess_valid']==0 for r in cd))
        ac=[r for r in summaries if (r['model_a'],r['model_b'])==('a','c')]
        self.assertTrue(all(r['within_sides']==1 for r in ac))
        ab=[r for r in summaries if (r['model_a'],r['model_b'])==('a','b')]
        self.assertTrue(all(r['within_sides']==2 for r in ab))

if __name__=='__main__':unittest.main(verbosity=2)
