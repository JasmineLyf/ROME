"""Offline regression tests for the paper protocol and leakage boundaries."""
import io
from pathlib import Path
import sys
import tempfile
import tokenize
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import common
import train_answer_pretrain as answer
import train_detect as detect
from roleplay.generate import parse_answers


class PaperAlignmentTests(unittest.TestCase):
    def test_user_splits_and_targets(self):
        train, val, test = common.load_splits()
        self.assertEqual([len(train), len(val), len(test)], [5205, 1735, 1735])
        self.assertFalse(set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
        with tempfile.TemporaryDirectory() as directory:
            for name, ids in zip(['train', 'val', 'test'], [train, val, test]):
                (Path(directory) / f'{name}_uids.txt').write_text('\n'.join(map(str, ids)))
            self.assertEqual(common.load_splits(directory), (train, val, test))
        targets = common.answer_targets(common.load_answers(), train + val)
        self.assertTrue(np.isfinite(targets.to_numpy()).all())

    def test_priors_ignore_held_out_data(self):
        samples = common.load_answers()
        metadata = common.load_metadata()
        train, val, test = common.load_splits()
        baseline = common.compute_priors(samples, metadata, train)
        samples.loc[samples.uids.isin(val + test), common.QUESTION_COLS] = 999
        metadata.loc[val + test, 'type'] = 'ENTJ'
        changed = common.compute_priors(samples, metadata, train)
        for expected, actual in zip(baseline, changed):
            np.testing.assert_array_equal(expected, actual)
            self.assertTrue(np.isfinite(actual).all())
            self.assertGreaterEqual(actual.min(), 0)
            self.assertLessEqual(actual.max(), 1)

    def test_priors_match_equations(self):
        # Q1: uncertainty=1, separation=2; Q2: uncertainty=0, separation=0.
        rows = []
        for uid, values in [(0, [-1, 1]), (1, [1, 3])]:
            for value in values:
                row = {'uids': uid, **{q: 0 for q in common.QUESTION_COLS}}
                row['Q1'] = value
                rows.append(row)
        metadata = pd.DataFrame({'type': ['ISTP', 'ENFJ']}, index=[0, 1])
        imp, rel = common.compute_priors(pd.DataFrame(rows), metadata, [0, 1])
        np.testing.assert_allclose(imp[:2], [1, 0])
        np.testing.assert_allclose(rel[:2], [0, 1])

    def test_macro_f1_includes_both_classes(self):
        truth = np.tile(np.array([0, 0, 0, 1])[:, None], (1, 4))
        predicted = np.zeros_like(truth)
        scores, mean = common.macro_f1(truth, predicted)
        np.testing.assert_allclose(scores, [3 / 7] * 4)
        self.assertAlmostEqual(mean, 3 / 7)

    def test_joint_loss_weights_and_robust_regression(self):
        logits = torch.zeros(2, 4, requires_grad=True)
        predictions = torch.full((2, 60), 3.0, requires_grad=True)
        loss = detect.joint_loss(logits, torch.zeros_like(logits), predictions, torch.zeros_like(predictions))
        self.assertAlmostEqual(loss.item(), 2.5 + 0.05 * np.log(2), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_moe_weighting_masks_and_gradients(self):
        onehot = common.question_onehot()
        indices = {d: np.where(onehot[:, i])[0].tolist() for i, d in enumerate(common.DIMS)}
        importance = np.linspace(0, 1, 60, dtype=np.float32)
        uncertainty = np.linspace(1, 0, 60, dtype=np.float32)
        users = {0: np.ones(5, dtype=np.float32)}
        with patch.multiple(detect, ANSWER_PRETRAIN_NUM_EXPERTS=2, ANSWER_PRETRAIN_HIDDEN_DIM=8,
                            TRUNK_HIDDEN=8, HEAD_HIDDEN=4):
            model = detect.DetectModel(np.ones((60, 3), dtype=np.float32), onehot,
                                       importance, uncertainty, users, indices, pretrained_path=None)
        captured = {}
        handles = []
        for dim in common.DIMS:
            handles.append(model.fusions[dim].register_forward_pre_hook(
                lambda module, args, dim=dim: captured.update({dim: args[1].detach().clone()})))
        logits, predictions = model(torch.ones(1, 5))
        self.assertEqual(logits.shape, (1, 4))
        self.assertEqual(predictions.shape, (1, 60))
        for dim in common.DIMS:
            idx = indices[dim]
            expected = predictions[:, idx].detach() * torch.tensor(importance[idx] * (1 - uncertainty[idx]))
            torch.testing.assert_close(captured[dim], expected)
        detect.joint_loss(logits, torch.zeros_like(logits), predictions, torch.zeros_like(predictions)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        for handle in handles:
            handle.remove()

    def test_detection_needs_no_generated_answers(self):
        data = detect.MBTIDataset([7], pd.DataFrame({'type': ['INTJ']}, index=[7]),
                                  None, {7: np.ones(5, dtype=np.float32)})
        emb, labels, _ = data[0]
        self.assertEqual(emb.shape, (5,))
        torch.testing.assert_close(labels, torch.tensor([0., 1., 0., 1.]))

    def test_answer_dataset_only_expands_selected_users(self):
        users = {uid: np.full(2, uid, dtype=np.float32) for uid in [1, 2, 3]}
        targets = pd.DataFrame(0., index=[1, 2, 3], columns=common.QUESTION_COLS)
        data = answer.AnswerPretrainDataset([1, 2], users, targets, np.ones((60, 3)))
        self.assertEqual(len(data), 120)
        self.assertEqual(data[0][0][0], 1)
        self.assertEqual(data[60][0][0], 2)
        self.assertEqual(data[119][0].shape, (9,))

    def test_ask_parser_rejects_incomplete_and_invalid_responses(self):
        valid = '\n'.join(f'Q{i}: +2' for i in range(1, 61)) + '\nINTJ'
        self.assertEqual(parse_answers(valid), ('INTJ', [2] * 60))
        for invalid in [valid.replace('Q60: +2', ''), valid.replace('Q1: +2', 'Q1: 4'), valid + '\nQ1: 1']:
            with self.assertRaises(ValueError):
                parse_answers(invalid)

    def test_two_stage_training_smoke(self):
        from contextlib import ExitStack, redirect_stdout, redirect_stderr
        train, val, test = common.load_splits()
        train, val, test = train[:32], val[:8], test[:8]
        users = {uid: np.random.default_rng(uid).normal(size=5).astype(np.float32)
                 for uid in train + val + test}
        samples = common.load_answers()
        samples = samples[samples.uids.isin(train + val)]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            directory = Path(directory)
            (directory / 'embeddings').mkdir()
            q_path = directory / 'embeddings/question_embeddings.npy'
            np.save(q_path, np.ones((60, 3), dtype=np.float32))
            for module in [answer, detect]:
                stack.enter_context(patch.object(module, 'load_splits', return_value=(train, val, test)))
                stack.enter_context(patch.object(module, 'load_user_embeddings', return_value=users))
                stack.enter_context(patch.object(module, 'load_answers', return_value=samples))
                stack.enter_context(patch.multiple(module, ANSWER_PRETRAIN_NUM_EXPERTS=2,
                                                 ANSWER_PRETRAIN_HIDDEN_DIM=8, DEVICE=torch.device('cpu')))
            stack.enter_context(patch.object(answer, 'DATA_DIR', directory))
            stack.enter_context(patch.multiple(detect, QUEST_EMB=q_path, TRUNK_HIDDEN=8, HEAD_HIDDEN=4))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            checkpoint = answer.train_answer_pretrain(42, 1, directory / 'answer.pth')
            result = detect.train_detect(42, 1, checkpoint, directory / 'detect.pth', directory / 'metrics.json')
            self.assertTrue(checkpoint.exists())
            self.assertTrue((directory / 'metrics.json').exists())
            self.assertTrue(0 <= result['average_macro_f1'] <= 1)

    def test_zero_epochs_cannot_reuse_stale_checkpoints(self):
        with self.assertRaisesRegex(ValueError, 'epochs must be positive'):
            answer.train_answer_pretrain(epochs=0)
        with self.assertRaisesRegex(ValueError, 'epochs must be positive'):
            detect.train_detect(epochs=0)

    def test_duplicate_embedding_posts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for split in ['train', 'test']:
                np.save(directory / f'{split}_post_embeddings.npy', np.ones((1, 3)))
                np.save(directory / f'{split}_post_index_map.npy', np.array([[1, 0]]))
            with self.assertRaisesRegex(ValueError, 'Duplicate post'):
                common.load_user_embeddings(directory)

    def test_fractional_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.csv'
            pd.DataFrame({'uids': [1.5], 'type': ['INTJ']}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'finite integers'):
                common.load_metadata(path)

    def test_missing_trials_warn_and_latest_repair_wins(self):
        rows = [{'user_id': '1_temp0.6', 'uids': 1, **dict.fromkeys(common.QUESTION_COLS, 1.)},
                {'user_id': '1_temp0.6', 'uids': 1, **dict.fromkeys(common.QUESTION_COLS, 2.)}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'answers.csv'
            pd.DataFrame(rows).to_csv(path, index=False)
            samples = common.load_answers(path)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples.iloc[0]['Q1'], 2)
            rows[1]['Q1'] = np.nan
            samples = pd.DataFrame(rows)
            with self.assertWarnsRegex(RuntimeWarning, 'missing Ask scores'):
                targets = common.answer_targets(samples, [1])
            self.assertEqual(targets.loc[1, 'Q1'], 1.)

    def test_ask_generation_resumes_failures_and_excludes_test_users(self):
        from contextlib import ExitStack, redirect_stdout
        from types import SimpleNamespace
        from roleplay import generate
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='\n'.join(f'Q{i}: 1' for i in range(1, 61)) + '\nINTJ'))])
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError('Simulated API failure')
            return response
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        metadata = pd.DataFrame({'type': ['INTJ', 'ENFP', 'ISTP'],
                                 'posts': ['train-text', 'val-text', 'never-test-text']}, index=[0, 1, 2])
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / 'answers.csv'
            stack.enter_context(patch.dict(sys.modules, {'openai': SimpleNamespace(OpenAI=lambda: fake_client)}))
            stack.enter_context(patch.object(sys, 'argv', ['generate.py', '--output', str(output)]))
            stack.enter_context(patch.object(generate, 'load_splits', return_value=([0], [1], [2])))
            stack.enter_context(patch.object(generate, 'load_metadata', return_value=metadata))
            stack.enter_context(patch.object(generate, 'load_questions', return_value=['Question'] * 60))
            stack.enter_context(patch.object(generate.time, 'sleep'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaisesRegex(RuntimeError, '1 Ask trials failed'):
                generate.main()
            self.assertEqual(len(pd.read_csv(output)), 9)
            generate.main()
            self.assertEqual(len(pd.read_csv(output)), 10)
            self.assertEqual(len(calls), 11)
            for call in calls:
                self.assertNotIn('never-test-text', call['messages'][0]['content'])
                self.assertEqual(call['model'], 'gpt-4o-2024-08-06')

    def test_no_chinese_comments(self):
        for path in (common.ROOT / 'scripts').rglob('*.py'):
            for token in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
                if token.type == tokenize.COMMENT:
                    self.assertFalse(any('\u4e00' <= c <= '\u9fff' for c in token.string), str(path))


if __name__ == '__main__':
    unittest.main()
