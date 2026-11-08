import json
from pathlib import Path
import tempfile
import unittest
import psutil
from slstack.lease import (AdmissionRefused, recover_runtime_leases,
                           release_runtime_cooperative)


class RuntimeLeaseRecoveryTests(unittest.TestCase):
    def fixture(self):
        root=Path(tempfile.mkdtemp(prefix='stack-lease-recovery-'))
        hold=root/'gpu.holds';hold.write_text('fixture ledger\n')
        current=psutil.Process()
        def record(location,identity,task='CUSTOM',live=False,uncertain=False,cooperative=False):
            directory=location/'state/leases';directory.mkdir(parents=True,exist_ok=True)
            row=dict(id=identity,task=task,hold_file=str(hold),released=False,
                     guardian_pid=current.pid if live else 99999999,
                     guardian_created=current.create_time() if live else 0,
                     cleanup_uncertain=uncertain)
            if cooperative:row['cooperative_retirement']=True
            path=directory/(identity+'.json');path.write_text(json.dumps(row));return path
        return root,hold,record

    def test_custom_owner_recovers_both_runtimes_and_preserves_foreign_live_and_uncertain(self):
        root,hold,record=self.fixture()
        own=record(root,'own');nested=record(root/'single-engine','nested')
        protected=[record(root,'foreign',task='OTHER'),record(root,'live',live=True),
                   record(root/'single-engine','uncertain',uncertain=True),
                   record(root,'coopdead',cooperative=True),
                   record(root,'cooplive',live=True,cooperative=True)]
        before={p:p.read_bytes() for p in protected}
        self.assertEqual(set(recover_runtime_leases(root,'CUSTOM')),{'own','nested'})
        self.assertTrue(json.loads(own.read_text())['recovered'])
        self.assertTrue(json.loads(nested.read_text())['recovered'])
        # A cooperative hold is evidence, not noise: an interrupted retirement may still
        # leave the producer holding the card, so recovery never releases it by inference.
        self.assertEqual(before,{p:p.read_bytes() for p in protected})
        self.assertNotIn('coopdead',hold.read_text())
        self.assertEqual(recover_runtime_leases(root,'CUSTOM'),[])

    def test_operator_release_needs_a_free_card_and_only_takes_dead_cooperative_holds(self):
        root,hold,record=self.fixture()
        dead=record(root,'coopdead',cooperative=True)
        kept=[record(root,'cooplive',live=True,cooperative=True),record(root,'plain'),
              record(root/'single-engine','uncertain',uncertain=True)]
        before={p:p.read_bytes() for p in kept}
        with self.assertRaises(AdmissionRefused):
            release_runtime_cooperative(root,'CUSTOM',holders=lambda:[dict(pid=1,used_memory_mib='1')])
        self.assertEqual(before,{p:p.read_bytes() for p in kept})
        self.assertEqual(json.loads(dead.read_text())['released'],False)
        self.assertEqual(release_runtime_cooperative(root,'CUSTOM',holders=lambda:[]),['coopdead'])
        self.assertIn('operator released dead cooperative guardian',hold.read_text())
        self.assertTrue(json.loads(dead.read_text())['operator_released'])
        self.assertEqual(before,{p:p.read_bytes() for p in kept})
        self.assertEqual(release_runtime_cooperative(root,'CUSTOM',holders=lambda:[]),[])


if __name__=='__main__':unittest.main()
