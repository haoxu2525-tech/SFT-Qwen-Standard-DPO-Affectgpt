"""CPU contract tests; real processor/video scanning is separate evidence."""
import ast
import contextlib
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
tree = ast.parse((ROOT/'dpo_interface.py').read_text(encoding='utf-8'))
names = {'read_pairs','membership_report','measure_pair','PairData'}
ns = dict(Path=Path, gzip=gzip, json=json, contextlib=contextlib, time=time,
          add_sft_path=lambda: None)
exec(compile(ast.Module(body=[n for n in tree.body if getattr(n,'name',None) in names],
                        type_ignores=[]),'dpo_interface.py','exec'),ns)

class NativeInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.row = dict(id='a',video='a.mp4',prompt='question',chosen='yes',rejected='no')
        self.tokenizer = lambda text, **kwargs: {'input_ids':list(range(len(text)))}
        self.encode = lambda *args: dict(input_ids=SimpleNamespace(shape=(1,20)),use_audio_in_video=True)

    def test_subset_keeps_excluded_ids_and_rejects_unknown_duplicates(self):
        result=ns['membership_report']([self.row],['a','b','c'])
        self.assertEqual(result,dict(original_count=3,selected_count=1,excluded_ids=['b','c']))
        for rows in ([dict(self.row,id='unknown')],[self.row,self.row],[]):
            with self.assertRaises(ValueError):ns['membership_report'](rows,['a'])

    def test_export_directory_only_reads_checksummed_successful_shards(self):
        p=self.root/'successful.jsonl.gz'
        p.write_bytes(gzip.compress((json.dumps(self.row)+'\n').encode()))
        (self.root/'failed.jsonl').write_text(json.dumps(dict(self.row,id='b',rejected='')))
        manifest=dict(files=[dict(kind='successful',name=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()),
                             dict(kind='failed',name='failed.jsonl')])
        (self.root/'manifest.json').write_text(json.dumps(manifest))
        self.assertEqual(ns['read_pairs'](self.root),[self.row])
        p.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):ns['read_pairs'](self.root)

    def test_chosen_cannot_change_independently_of_source_answer(self):
        p=self.root/'pairs.jsonl'
        p.write_text(json.dumps(dict(self.row,ground_truth_answer='different')))
        with self.assertRaises(ValueError):ns['read_pairs'](p)

    def test_overflow_is_measured_without_clipping_or_losing_row(self):
        info,error=ns['measure_pair'](self.encode,None,self.tokenizer,self.root,('train',self.row),10,9)
        self.assertEqual((info['prompt'],info['total'],info['chosen_after']),(20,24,4))
        self.assertEqual(error['kind'],'length_limit')

    def test_answer_overflow_also_fails(self):
        info,error=ns['measure_pair'](self.encode,None,self.tokenizer,self.root,('train',self.row),22,21)
        self.assertEqual(error['kind'],'length_limit')
        self.assertEqual(info['chosen_before'],info['chosen_after'])

    def test_false_audio_flag_and_path_escape_fail(self):
        wrong=lambda *args:dict(input_ids=SimpleNamespace(shape=(1,5)),use_audio_in_video=False)
        for encode,row in [(wrong,self.row),(self.encode,dict(self.row,video='../escape.mp4'))]:
            info,error=ns['measure_pair'](encode,None,self.tokenizer,self.root,('train',row),100,99)
            self.assertIsNone(info)
            self.assertEqual(error['kind'],'encoding_error')

    def test_pair_route_uses_shared_encoder_and_ignores_api_metadata(self):
        with patch.dict(sys.modules,{'omni_components':SimpleNamespace(encode_prompt=self.encode)}):
            data=ns['PairData']([dict(self.row,annotation_api={'reasoning_content':'private'},audio='missing.wav')],[],None,self.root,None)
        self.assertIs(data.encode,self.encode)
        route,pair=next(iter(data.get_dataset('affectgpt_preferences','train').items()))
        self.assertEqual(set(json.loads(route)),{'id','prompt','video'})
        self.assertEqual(pair['responses'],['yes','no'])
        self.assertEqual(pair['pairs'],[(0,1)])

    def test_preflight_records_all_rows_before_raising(self):
        with patch.dict(sys.modules,{'omni_components':SimpleNamespace(encode_prompt=self.encode)}):
            data=ns['PairData']([self.row,dict(self.row,id='b')],[],None,self.root,None)
        p=self.root/'report.json'
        with self.assertRaises(ValueError):data.preflight(self.tokenizer,10,9,p)
        report=json.loads(p.read_text())
        self.assertTrue(report['complete'])
        self.assertEqual(report['processed'],2)
        self.assertEqual(report['summary']['train']['max_prompt'],20)
        self.assertEqual(report['over_max_length_rows'],2)
        self.assertEqual(report['truncated_rows'],0)

if __name__=='__main__':unittest.main()
