import importlib.util, json, sys, tempfile, unittest
from pathlib import Path
from datetime import datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import store, dashboard_data, telemetry, model_registry

class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.old=store.ROOT;store.ROOT=Path(self.tmp.name);store.initialize()
        self.now=datetime(2026,11,11,15,tzinfo=store.TZ)
        self.start=datetime(2026,11,11,0,tzinfo=store.TZ).timestamp()
    def tearDown(self):
        store.ROOT=self.old;self.tmp.cleanup()
    def test_boundary_and_non_destructive(self):
        with store.database() as db:
            db.executemany('INSERT INTO intervals VALUES (?,?,?,?)',[
                ('person','sit',self.start-600,self.start+600),('person','sit',self.start+600,self.start+1200),
                ('person','privacy',self.start+1200,self.start+1500),('person','sit',self.start+1500,self.start+1800),
                ('presence','present',self.start,self.start+1200)])
        before=(store.ROOT/'data.sqlite3').read_bytes()
        result=dashboard_data.dashboard_snapshot('2026-11-11',self.now)
        self.assertEqual(result['totals'],{'sit':1500,'privacy':300,'present':1200})
        self.assertEqual(len(result['intervals']),4)
        self.assertEqual(sum(r['sit'] for r in result['hourly']),1500)
        self.assertEqual(before,(store.ROOT/'data.sqlite3').read_bytes())
        self.assertEqual(dashboard_data.dashboard_snapshot('2026-11-10',self.now)['totals']['sit'],600)
    def test_notes_consecutive_only(self):
        rows=[dict(kind='person',state='sit',start=self.start,end=self.start+200),dict(kind='person',state='privacy',start=self.start+200,end=self.start+300),dict(kind='person',state='sit',start=self.start+300,end=self.start+1000)]
        notes=[dict(id=i,kind='cosmos',body=b,at=self.start+t) for i,(t,b) in enumerate([(10,'坐'),(20,'坐'),(310,'坐'),(320,'起身'),(330,'坐'),(340,'坐')])]
        result=dashboard_data.group_notes(notes,rows)
        self.assertEqual([r['count'] for r in result],[2,1,1,2])
        self.assertEqual(sum(r['count'] for r in result),len(notes))
    def test_unknown_and_gaps_never_bridged(self):
        rows=[dict(kind='person',state=s,start=a,end=b) for s,a,b in [('sit',0,10),('sit',11,20),('unknown',20,25),('sit',25,30)]]
        self.assertEqual(len(dashboard_data.merge_intervals(rows)),4)
    def test_empty_future_invalid(self):
        self.assertEqual(dashboard_data.dashboard_snapshot('2026-11-12',self.now)['totals'],{})
        with self.assertRaises(ValueError):dashboard_data.dashboard_snapshot('bad',self.now)
    def test_registry_allowlist_and_persistence(self):
        (store.ROOT/'models').mkdir();(store.ROOT/'models/ggml-base.bin').write_bytes(b'test')
        with self.assertRaises(ValueError): model_registry.select_asr('../../tmp/x')
        with self.assertRaises(ValueError): model_registry.select_asr('turbo')
        model_registry.select_asr('base');self.assertEqual(model_registry.current(),'base')
        self.assertEqual(model_registry.asr_path(),store.ROOT/'models/ggml-base.bin')
    def test_human_annotation_preserves_originals_and_total(self):
        start=self.start+14*3600
        with store.database() as db:
            db.execute('INSERT INTO intervals VALUES (?,?,?,?)',('person','unknown',start,start+3600))
            db.execute('INSERT INTO intervals VALUES (?,?,?,?)',('presence','unknown',start,start+3600))
        store.annotate_activity(start+600,start+1200,'away','user confirmed at minute precision')
        result=dashboard_data.dashboard_snapshot('2026-11-11',self.now)
        self.assertEqual(result['totals']['away'],600)
        self.assertEqual(result['totals']['unknown'],3000)
        self.assertEqual(sum(v for k,v in result['hourly'][14].items() if k!='hour'),3600)
        self.assertTrue(any(r.get('source')=='manual' for r in result['intervals']))
        with store.database() as db:
            self.assertEqual([r['state'] for r in db.execute('SELECT * FROM intervals')],['unknown','unknown'])
        # A newer annotation supersedes only its own overlap, without doubling duration.
        store.annotate_activity(start+900,start+1200,'sit','newer label')
        result=dashboard_data.dashboard_snapshot('2026-11-11',self.now)
        self.assertEqual(result['totals']['away'],300)
        self.assertEqual(result['totals']['sit'],300)
        self.assertEqual(result['totals']['present'],300)

    def test_privacy_not_rewritten_by_activity_annotation(self):
        with store.database() as db:
            db.execute('INSERT INTO intervals VALUES (?,?,?,?)',('person','privacy',self.start,self.start+60))
        store.annotate_activity(self.start,self.start+60,'away','user label')
        result=dashboard_data.dashboard_snapshot('2026-11-11',self.now)
        self.assertEqual(result['totals'],{'privacy':60})

    def test_tegrastats(self):
        result=telemetry.parse_tegrastats('CPU [4%@729,off] GR3D_FREQ 99% VDD_GPU_SOC 7185mW/7185mW VIN_SYS_5V0 4807mW/4807mW')
        self.assertEqual(result['gpu_pct'],99);self.assertEqual(result['cpu_cores'],[4,None]);self.assertEqual(result['power_rails_w']['VDD_GPU_SOC'],7.185)
        self.assertNotIn('gpu_pct',telemetry.parse_tegrastats('unavailable'))

if __name__ == '__main__':
    unittest.main(verbosity=2)
