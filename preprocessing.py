'''
Preprocessing functions for text and audio records.

The input to each function includes:
- `r`, the record to be preprocessed
- `src_or_target`: a string, either 'source' or 'target', specifying which part
    of the record to modify.
- Any other keyword arguments defined in the config.

Typical format of a record:

{
  'source': 'Some text',
  'source.language': 'eng',
  'source.origin_dataset': 'salt',
  'target': [0.0, 0.0, ...],
  'target.sample_rate': 16000,
  'target.language': 'lug',
  'target.is_studio': False,
}

The keys depend on whether the source and target are text or audio respectively.

If the @single_batch_entry decorator is removed, then a whole batch is passed
to each function at once. The structure of the input is then e.g.:

{
  'source': ['Text of entry 1', 'Text of entry 2', ...],
  'source.language': ['eng', 'eng', ...]
  ...
}
'''

import re
import string
import random
import cleantext
import functools
import numpy as np
import librosa
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.audio as naa

from .utils import single_batch_entry



@single_batch_entry
def prefix_dataset_tag(r, src_or_tgt, tags=None):
    '''If the origin dataset name matches a string, prefix a tag to the text.'''
    for origin, tag in tags.items():
        if origin in r[f'{src_or_tgt}.origin_dataset']:
            r[src_or_tgt] = tag + ' ' + r[src_or_tgt]
    return r

@single_batch_entry
def prefix_target_language(r, src_or_tgt):
    r[src_or_tgt] = f'>>{r["target.language"]}<< ' + r[src_or_tgt]
    return r

@single_batch_entry
def match_target_sentence_format_to_source(r, src_or_tgt):
    '''Match the sentence formatting of the target text to the source text.
    
    Sets the capitalisation of the first letter in the target text, and
    the presence of a trailing full stop, to match whatever is in the source
    text.'''
    if r['source'][0].isupper():
        r['target'] = r['target'][0].upper() + r['target'][1:]
    else:
        r['target'] = r['target'][0].lower() + r['target'][1:]
        
    source_has_full_stop = (r['source'][-1] == '.')
    target_has_full_stop = (r['target'][-1] == '.')
    
    if source_has_full_stop and not target_has_full_stop:
        r['target'] = r['target'] + '.' 
        
    if target_has_full_stop and not source_has_full_stop:
        r['target'] = r['target'][:-1]
        
    return r

@single_batch_entry
def clean_text(r, src_or_tgt, **clean_text_args):
    r[src_or_tgt] = cleantext.clean(
        r[src_or_tgt], to_ascii=False, lower=False, **clean_text_args)
    return r

def augment_characters(r, src_or_tgt, **char_augmentation_params):
    char_augmenter = nac.RandomCharAug(**char_augmentation_params)
    r[src_or_tgt] = char_augmenter.augment(r[src_or_tgt])
    return r

def augment_words(r, src_or_tgt, **word_augmentation_params):
    word_augmenter = naw.RandomWordAug(**word_augmentation_params)
    r[src_or_tgt] = word_augmenter.augment(r[src_or_tgt])
    return r

@single_batch_entry
def augment_audio_noise(r, 
                        src_or_tgt,
                        max_relative_amplitude = .5,
                        max_coverage = .6,
                        min_coverage = .1):
    '''Add random noise to an audio sample.
    
    Args:
        r: dictionary containing fields of a single dataset row.
        src_or_tgt: str, key such that r[src_or_tgt] contains the audio array
            to be augmented.
        max_relative_amplitude: max noise amplitude relative to the largest
            value in the source array x. The value chosen is uniform in the
            range (0, max_amplitude_relative).
        max_coverage: largest proportion of the audio sample that will have
            noise added. A value of 1 means that potentially the entire sample
            can have noise added.
        min_coverage: smallest proportion of the audio sample that can have
            noise added. The coverage for a particular sample is randomly chosen
            in the range (min_coverage, max_coverage), and a segment of this
            length is randomly selected from the sample.
    '''
    
    x = r[src_or_tgt]
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    x_max = np.amax(np.abs(x))
    amplitude = np.random.uniform(0, max_relative_amplitude) * x_max
    coverage = np.random.uniform(min_coverage, max_coverage)
    num_samples_to_affect = int(len(x) * coverage)
    start_index = np.random.randint(0, len(x) - num_samples_to_affect)

    # Generate noise of the determined amplitude
    noise = np.random.uniform(0, amplitude, size=num_samples_to_affect)

    # Apply noise to the chosen segment
    x_with_noise = np.copy(x)  # Make a copy of x to prevent altering the original
    x_with_noise[start_index:start_index + num_samples_to_affect] += noise
    
    r[src_or_tgt] = x_with_noise
    return r

@single_batch_entry
def augment_audio_time_masking(r,
                               src_or_tgt,
                               num_masks_min=2,
                               num_masks_max=4,
                               max_mask_duration_ms=100):
    """Apply time masking to an audio signal, with cutouts of random duration.

    Args:
        r: dictionary containing fields of a single dataset row.
        src_or_tgt: str, key such that r[src_or_tgt] contains the audio array
            to be augmented.
        num_masks_min, num_masks_max: int, the range of masked periods is
            randomly chosen in this range.
        max_mask_duration_ms: int, the maximum duration of a mask in
            milliseconds.
    """
    audio_masked = np.copy(r[src_or_tgt]) # Avoid modifying the original
    
    # Convert maximum mask duration from milliseconds to number of samples
    sample_rate = r[f'{src_or_tgt}.sample_rate']
    max_mask_duration_samples = int((sample_rate / 1000) * max_mask_duration_ms)
    
    num_masks = np.random.randint(num_masks_min, num_masks_max)
    total_time_steps = len(audio_masked)
    
    for _ in range(num_masks):
        mask_duration = np.random.randint(0, max_mask_duration_samples + 1)  
        t0 = np.random.randint(0, total_time_steps - mask_duration + 1)
        audio_masked[t0:t0 + mask_duration] = 0

    r[src_or_tgt] = audio_masked
    return r

@single_batch_entry
def clean_and_remove_punctuation(
    r, src_or_tgt, allowed_punctuation=None, **clean_text_args):
    r[src_or_tgt] = cleantext.clean(
        r[src_or_tgt], to_ascii=False, no_punct=False, **clean_text_args)
    
    punct = list(string.punctuation)
    if allowed_punctuation:
        for allowed in allowed_punctuation:
            punct.remove(allowed)
        
    r[src_or_tgt] = ''.join([c for c in r[src_or_tgt] if c not in punct])
    return r
    
@single_batch_entry
def lower_case(r, src_or_tgt):
    r[src_or_tgt] = r[src_or_tgt].lower()
    return r

@single_batch_entry
def random_capitalise_source_and_target(r, src_or_tgt, p=0.005):
    if np.random.random() < p:
        r['source'] = r['source'].upper()
        r['target'] = r['target'].upper()
    return r

@single_batch_entry
def set_sample_rate(r, src_or_tgt, rate):
    '''Resamples audio data, if the sample rate in the record is different.'''
    current_sample_rate = r[f'{src_or_tgt}.sample_rate']
    if current_sample_rate != rate:
        audio_data = np.array(r[src_or_tgt])
        resampled_audio_data = librosa.resample(
            audio_data, orig_sr=current_sample_rate, target_sr=rate)
        r[src_or_tgt] = resampled_audio_data
        r[f'{src_or_tgt}.sample_rate'] = rate
    
    return r

# TODO: Check that the order of preprocessing operations makes sense. For
# example, don't call match_target_sentence_format_to_source after 
# prefix_dataset_tag (because then the tag is part of the text)
    