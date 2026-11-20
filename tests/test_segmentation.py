import json
import pickle
import random
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

from neseg.prepare import make_rows
from neseg.protocol import parse_action
from neseg.reward import compute_score, encode, point_scores
from neseg.rollout import play
from preprocess_data.preprocess import process_rvos, sample_frame_indices, sample_record
from preprocess_data.ref_sav import ref_sav_samples


class SegmentationTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((11, 11), dtype=bool)
        self.mask[3:8, 3:8] = True
        self.action = dict(action='segment', time=0, positive_points=[[500, 500]], negative_points=[[800, 500]])
        self.refs = [[500, 500]]

    def test_point_signs_distance_and_dino_gate(self):
        grid = torch.zeros((11, 11, 2))
        grid[:, :, 0] = 1
        grid[5, 9] = torch.tensor([0., 1.])
        good = point_scores(self.action, self.mask, self.refs, grid)
        self.assertEqual(list(good.values()), [1., 1., 1., 1.])
        wrong = dict(self.action, positive_points=[[0, 0]], negative_points=[[500, 500]])
        bad = point_scores(wrong, self.mask, self.refs, grid)
        self.assertEqual((bad['positive_hit'], bad['negative_hit'], bad['negative_dino']), (0., -1., 0.))
        self.assertLess(bad['positive_distance'], good['positive_distance'])
        easy = point_scores(dict(self.action, negative_points=[[900, 500]]), self.mask, self.refs, grid)
        self.assertEqual(easy['negative_dino'], 0.)

    def test_reward_uses_final_video_mask_and_actual_call_count(self):
        rle = encode(self.mask)
        truth = json.dumps(dict(masks=[rle, None], references=[self.refs, []], features=''))
        call = dict(self.action, frame=0)
        trace = dict(calls=[call], masks=[rle, rle], valid=True)
        first = compute_score(trace, truth, negative_dino_weight=0)
        self.assertEqual(first['iou'], .5)
        second = compute_score(dict(trace, calls=[call, call]), truth, negative_dino_weight=0)
        self.assertLess(second['efficiency'], first['efficiency'])
        self.assertEqual(compute_score(dict(trace, valid=False), truth)['overall'], 0.)
        with self.assertRaisesRegex(ValueError, 'DINOv2'):
            compute_score(trace, truth)

    def test_rollout_consumes_tool_feedback_and_masks_it_from_loss(self):
        mask = self.mask
        observed = []
        class Tool:
            def apply(self, action):
                observed.append(action['frame'])
                return [mask, mask]
        @contextmanager
        def open_tool(_):
            yield Tool()
        class Tokens:
            def convert_tokens_to_ids(self, _):
                return 0
            def encode(self, text, **_):
                return list(text.encode())
            def decode(self, tokens, **_):
                return bytes(tokens).decode()
        codec = Tokens()
        def tokens(action):
            return codec.encode(json.dumps(action)) + [0]
        further = iter([dict(self.action, time=1), {'action': 'finish'}])
        def generate(history, remaining):
            self.assertIn(b'SAM 2 mask', bytes(history))
            return tokens(next(further))
        args = dict(media={'times': [0, 1], 'images': []}, tokenizer=codec, generate=generate,
                    budget=4096, max_calls=3, open_session=open_tool)
        ids, policy, trace = play(tokens(self.action), **args)
        self.assertTrue(trace['valid'])
        self.assertEqual(observed, [0, 1])
        self.assertNotIn(b'SAM 2 mask', bytes(t for t, p in zip(ids, policy) if p))
        self.assertEqual(len(trace['masks']), 2)
        self.assertFalse(play(tokens({'action': 'finish'}), **args)[2]['valid'])
        self.assertTrue(play(tokens(self.action), **dict(args, max_calls=1))[2]['valid'])
        with self.assertRaises(ValueError):
            parse_action(json.dumps(dict(self.action, time=.5)), [0, 1])

    def test_rvos_uses_source_expression_order_and_frame_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp, 'video_datas', 'rvos')
            meta = base / 'meta_expressions/train'
            frames = base / 'train/JPEGImages/clip'
            meta.mkdir(parents=True)
            frames.mkdir(parents=True)
            for name in ('00000', '00001'):
                cv2.imwrite(str(frames / (name + '.jpg')), np.zeros((11, 11, 3), dtype=np.uint8))
            video = dict(frames=['00001', '00000'], expressions={'2': {'exp': 'second'}, '10': {'exp': 'first'}})
            (meta / 'meta_expressions.json').write_text(json.dumps({'videos': {'clip': video}}))
            with (base / 'mask_dict.pkl').open('wb') as stream:
                pickle.dump({'0': [encode(self.mask), None], '1': [None, encode(self.mask)]}, stream)
            first, second = process_rvos(tmp, 'train', 8, fps=4)
            self.assertEqual((first['expression'], second['expression']), ('first', 'second'))
            self.assertEqual(first['video_meta']['sampled_times'], [0., .25])
            self.assertIsNone(second['frame_masks']['1'][0])
            self.assertTrue(first['frame_paths'][0].endswith('00000.jpg'))

    def test_ref_sav_alignment_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root/'video_datas/sam_v_full/sav_train/shard'
            assets.mkdir(parents=True)
            path = assets/'v.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 24, (32, 32))
            self.assertTrue(writer.isOpened())
            for i in range(9):
                writer.write(np.full((32, 32, 3), i * 22, dtype=np.uint8))
            writer.release()
            masks = np.zeros((32, 32), dtype=bool)
            masks[4:18, 2:20] = True
            rle = encode(masks)
            (assets/'anno.json').write_text(json.dumps(dict(masklet=[[rle]]*3)))
            item = dict(video_path='shard/sav_train/shard/v.mp4', anno_path='shard/sav_train/shard/anno.json',
                        objects={'0': {'formated': 'the rectangle'}})
            (root/'video_datas/Ref-SAV.json').write_text(json.dumps({'v': item}))
            raw, = ref_sav_samples(root, 8, sample_frame_indices)
            self.assertEqual(raw['video_meta']['sampled_frame_ids'], [0, 4, 8])
            self.assertGreater(cv2.imread(raw['frame_paths'][-1]).mean(), 160)
            rng = random.Random(42)
            rec = sample_record(problem_id=1, data_root=str(root), feature_tensor=None,
                                feature_output_path=None, rng=rng, sft_tool_calls=1, **raw)
            sft, rl = make_rows(rec, root, rng)
            self.assertEqual(len(sft), 1)
            self.assertEqual(rl['problem'].count('<image>'), 3)
            self.assertNotIn('counts', rl['problem'])
            self.assertEqual(len(json.loads(rl['answer'])['masks']), 3)
            self.assertTrue(parse_action(sft[0]['target'], rl['neseg_times']))


if __name__ == '__main__':
    unittest.main()
