import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import evaluate

from .metrics import multilingual_eval, multilingual_eval_fn

# Helper function to create a mock tokenizer
def create_mock_tokenizer():
    mock_tokenizer = MagicMock()
    mock_tokenizer.batch_decode = MagicMock(
        side_effect=(lambda x, skip_special_tokens:
            [' '.join([str(i) for i in r]) for r in x]))
    mock_tokenizer.pad_token_id = 0
    return mock_tokenizer

# Define our unit test case
class MultilingualEvalUnitTest(unittest.TestCase):
    
    def test_multilingual_eval(self):
        mock_tokenizer = create_mock_tokenizer()
        metric = evaluate.load('sacrebleu')

        predictions = np.array([[1, 2, 3, 4], [4, 5, 6, 6]])
        labels = np.array([[1, 2, 3, 4], [4, 5, 3, 6]])

        eval_preds = (predictions, labels)
        source_language = ['lug', 'ach']
        target_language = ['nyn', 'teo']
        metrics = [metric]
        metric_names = ['BLEU']
        log_first_N_predictions = 0
        
        result = multilingual_eval(eval_preds,
                                   source_language,
                                   target_language,
                                   metrics,
                                   metric_names,
                                   mock_tokenizer,
                                   log_first_N_predictions)

        # Assert: Check if the output matches the expected result
        self.assertAlmostEqual(result['BLEU_lug_nyn'], 100.0)
        self.assertAlmostEqual(result['BLEU_ach_teo'], 35.355)
        # Assert function calls of tokenizer
        mock_tokenizer.batch_decode.assert_called()
        
    def test_create_metrics_fn(self):
        
        mock_tokenizer = create_mock_tokenizer()

        predictions = np.array([[1, 2, 3, 4], [4, 5, 6, 6]])
        labels = np.array([[1, 2, 3, 4], [4, 5, 3, 6]])

        eval_preds = (predictions, labels)
        source_language = ['lug', 'ach']
        target_language = ['nyn', 'teo']
        eval_dataset = {
            'source.language': source_language,
            'target.language': target_language,
        }
        
        metrics = [evaluate.load('sacrebleu'),
                   evaluate.load('wer'),
                   evaluate.load('cer'),
                  ]
        
        compute_metrics = multilingual_eval_fn(
            eval_dataset,
            metrics,
            mock_tokenizer,
        )
        
        result = compute_metrics(eval_preds)

        self.assertAlmostEqual(result['BLEU_lug_nyn'], 100.0)
        self.assertAlmostEqual(result['BLEU_ach_teo'], 35.355)   
        self.assertAlmostEqual(result['CER_ach_teo'], 0.143)

if __name__ == '__main__':
    unittest.main()