import numpy as np
import pandas as pd
import functools
import numbers


def multilingual_eval(eval_preds,
                      source_language,
                      target_language,
                      metrics,
                      metric_names,
                      tokenizer,
                      log_first_N_predictions,
                      speech_processor=None):
    '''Compute metric scores for each source and target language combination.'''
    def _round_if_float(f, p):
        if isinstance(f, float):
            return round(f, p)
        else:
            return f

    if speech_processor:
        pred_logits = eval_preds.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        eval_preds.label_ids[eval_preds.label_ids == -
                             100] = speech_processor.tokenizer.pad_token_id
        decoded_predictions = speech_processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        decoded_labels = speech_processor.batch_decode(
            eval_preds.label_ids, group_tokens=False)
    else:
        predictions, labels = eval_preds
        # Replace -100 values as we can't decode them.
        predictions = np.where(
            predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

    if log_first_N_predictions:
        print('First N predictions in eval set:')
        for i in range(log_first_N_predictions):
            print(f'Prediction ({source_language[i]} to {target_language[i]}):'
                  f' "{decoded_predictions[i]}", '
                  f'True label: "{decoded_labels[i]}"')

    subsets = {}
    for i in range(len(decoded_predictions)):
        if speech_processor:
            # For speech metrics, such as WER, we evaluate for separate target
            # languages.
            language_combination = target_language[i]
        else:
            # For translation metrics, such as BLEU, we want metrics for every
            # source/target combination.
            language_combination = source_language[i] + \
                '_' + target_language[i]
        if language_combination not in subsets:
            subsets[language_combination] = {'predictions': [], 'labels': []}
        subsets[language_combination]['predictions'].append(
            decoded_predictions[i])
        subsets[language_combination]['labels'].append(decoded_labels[i])

    result = {}
    for metric, metric_name in zip(metrics, metric_names):
        for subset in list(subsets.keys()):
            result_subset = metric.compute(
                predictions=subsets[subset]['predictions'],
                references=subsets[subset]['labels'])
            if metric_name == 'BLEU':
                # The sacrebleu and bleu implementations have different formats
                r = result_subset.get('score') or result_subset.get('bleu')
            else:
                if not isinstance(result_subset, numbers.Number):
                    raise ValueError(
                        'Expected a metric that yields a single value, but the '
                        f'result from metric "{metric_name.lower()}" was '
                        f'{result_subset}. Supported metrics include '
                        'sacrebleu, wer, cer.')
                r = result_subset

            result[f'{metric_name}_{subset}'] = r

        subset_values = [result[f'{metric_name}_{subset}']
                         for subset in list(subsets.keys())]
        try:
            result[f'{metric_name}_mean'] = np.mean(subset_values)
        except TypeError:
            result[f'{metric_name}_mean'] = np.nan

    result = {k: _round_if_float(v, 3) for k, v in result.items()}
    return result


def multilingual_eval_fn(eval_dataset,
                         metrics,
                         tokenizer,
                         log_first_N_predictions=0,
                         speech_processor=None):
    '''Return a function with the signature `eval_fn(preds)`.

    If `speech_processor` is defined, then it is used to decode the predictions.
    Otherwise `tokenizer` is used (e.g. for translation tasks).'''

    df = pd.DataFrame(eval_dataset)
    source_language = list(df['source.language'])
    target_language = list(df['target.language'])

    metric_names = []
    for m in metrics:
        if m.name == 'sacrebleu':
            metric_names.append('BLEU')
        else:
            metric_names.append(m.name.upper())

    return lambda x: multilingual_eval(
        x,
        source_language,
        target_language,
        metrics,
        metric_names,
        tokenizer,
        log_first_N_predictions=log_first_N_predictions,
        speech_processor=speech_processor)
