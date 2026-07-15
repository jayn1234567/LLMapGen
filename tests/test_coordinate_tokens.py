import json
import unittest

from mllm.coordinate_tokens import (
    append_coordinate_token_instruction,
    decode_coordinate_tokens,
    encode_coordinate_conversations,
    encode_map_json_coordinates,
    normalize_coordinate_token_mode,
    register_coordinate_vocabulary,
)


class _Encoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _FakeTokenizer:
    def __init__(self):
        self.base_size = 32
        self.added_vocab = {}
        self.last_special_tokens = None

    def __len__(self):
        return self.base_size + len(self.added_vocab)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in self.added_vocab:
            return _Encoding([self.added_vocab[text]])
        return _Encoding([1, 2] if len(text) > 1 else [1])

    def add_tokens(self, tokens, special_tokens=False):
        self.last_special_tokens = special_tokens
        for token in tokens:
            if token not in self.added_vocab:
                self.added_vocab[token] = self.base_size + len(self.added_vocab)
        return len(tokens)

    def get_added_vocab(self):
        return dict(self.added_vocab)


class _FakeModel:
    def __init__(self):
        self.resized_to = None

    def resize_token_embeddings(self, size):
        self.resized_to = size


class CoordinateTokenTests(unittest.TestCase):
    def test_encode_and_decode_only_points_coordinates(self):
        payload = {
            "version": 956,
            "lines": [
                {
                    "category": "centerline",
                    "lane_type": 2,
                    "points": [[956, 4], [1000, 0]],
                },
                {
                    "category": "intersection",
                    "intersection_type": 1,
                    "points": [[0, 12], [25, 30], [0, 12]],
                },
            ],
        }
        encoded = encode_map_json_coordinates(json.dumps(payload))
        self.assertIn('"version":956', encoded)
        self.assertIn('"lane_type":2', encoded)
        self.assertIn('"points":[[<956>,<4>],[<1000>,<0>]]', encoded)
        self.assertNotIn('"<956>"', encoded)
        self.assertEqual(json.loads(decode_coordinate_tokens(encoded)), payload)

    def test_conversation_transform_changes_assistant_only(self):
        conversations = [
            {"from": "human", "value": "<image>\nCoordinates use 0-1000."},
            {
                "from": "gpt",
                "value": '{"lines":[{"category":"centerline","points":[[956,42],[1,2]]}]}',
            },
        ]
        encoded = encode_coordinate_conversations(conversations)
        self.assertIn("unquoted discrete coordinate token <n>", encoded[0]["value"])
        self.assertIn("[[<956>,<42>],[<1>,<2>]]", encoded[1]["value"])
        self.assertNotEqual(encoded[0]["value"], conversations[0]["value"])
        self.assertEqual(conversations[1]["value"].count("<956>"), 0)

    def test_instruction_is_idempotent(self):
        once = append_coordinate_token_instruction("Prompt")
        twice = append_coordinate_token_instruction(once)
        self.assertEqual(once, twice)

    def test_invalid_coordinate_fails_before_training(self):
        with self.assertRaisesRegex(ValueError, "outside 0-1000"):
            encode_map_json_coordinates('{"points":[[1001,0]]}')

    def test_mode_aliases(self):
        self.assertEqual(normalize_coordinate_token_mode("discrete"), "angle")
        self.assertEqual(normalize_coordinate_token_mode("off"), "none")

    def test_vocabulary_is_atomic_but_not_registered_as_control_tokens(self):
        tokenizer = _FakeTokenizer()
        model = _FakeModel()
        report = register_coordinate_vocabulary(tokenizer, model, max_coordinate=3)
        self.assertFalse(tokenizer.last_special_tokens)
        self.assertEqual(model.resized_to, 36)
        self.assertEqual(report["added_tokens"], 4)
        self.assertEqual(report["discrete_tokens_per_coordinate"], 1)


if __name__ == "__main__":
    unittest.main()
