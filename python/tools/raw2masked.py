#!/usr/bin/env python3
import argparse
import logging
import os
import simplejson

from random import random, seed
from time import time


def confirm_overwrite(fn):
    """Ask user to enter Y or N (case-insensitive) to confirm whether
file with specified name/path should be overwritten."""
    answer = ""
    while answer not in ['y', 'n']:
        answer = input(f'File {fn} exists. Overwrite [y] or abort [n]?')
        answer = answer.lower().strip()[0]
    return answer == 'y'


def flatten_record(jline, feature_ids):
    for parent in ['dkim', 'dmarc', 'spf']:
        if f'{parent}Dsbld' in jline:
            orig_val = jline.pop(f'{parent}Dsbld')
            jline[f'{parent}.enabled'] = int(not orig_val)
        if parent in jline:
            orig_val = jline.pop(parent)
            for field in orig_val:
                val = orig_val[field]
                # Replacing booleans with integers to safe space in json.
                new_val = int(val) if isinstance(val, type(False)) else val
                # Rename dkim.valid to dkim.pass for consistency with spf
                if parent == 'dkim' and field == 'valid':
                    field = 'pass'
                jline[f'{parent}.{field}'] = new_val
                # Add boolean indicators for whether spf passed (and where).
                if parent == 'spf' and field in ['fromResult', 'heloResult']:
                    if val == 'pass':
                        jline['spf.pass'] = 1  # 1 if either from or helo passed
                        jline[f'{parent}.{field}.pass'] = 1
    if 'response' in jline:
        orig_val = jline.pop('response')
        for resp in orig_val:
            if 'field' not in resp:
                continue
            context = resp['field']
            if 'domain' in resp:
                for field in resp['domain']:
                    jline[f'{context}.domain.{field}'] = resp['domain'][field]
            if 'features' in resp:
                features = set(resp['features'])
                jline[f'{context}.feature_list'] = sorted(features)
                for feature in feature_ids:
                    feat_present = 1 if feature in features else 0
                    jline[f'{context}.feature.{feature}'] = feat_present


if __name__ == "__main__":
    # Parse commandline and prepare for main-loop.
    parser = argparse.ArgumentParser(description=('Replace sensitive information, '
                                                  'add categorical variables, '
                                                  'and adjust field names, '
                                                  'such that the dataset can be '
                                                  'shared publicly.'))
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('-v', '--verbose',
                           help='Increase output verbosity.',
                           action='count', default=0)
    verbosity.add_argument('-q', '--quiet',
                           help='Quiet, create only minimal output.',
                           action='store_true')
    parser.add_argument('-i', '--input', required=True,
                        help=('Filename of raw data input data in '
                              'line-delimited JSON format.'))
    parser.add_argument('-p', '--prefix', required=False,
                        help=('Prefix for output files. Defaults to: '
                              '<input_dir>/<timestamp>_'))
    parser.add_argument('-r', '--reference', required=False,
                        help=('Path to file containing reference timestamp, '
                              'neutral and non-neutral counts as well as '
                              'feature- and domain maps in json format. '
                              'If specified, maps are updated, otherwise '
                              'maps are build from scratch.'))
    parser.add_argument('-u', '--unbalanced',
                        help=('Create unbalanced (with regards to verdict) '
                              'output in addition to stratified, balanced.'),
                        action='store_true')
    args = parser.parse_args()

    # Set output verbosity.
    if args.quiet:
        loglevel = logging.CRITICAL
    elif args.verbose == 0:
        loglevel = logging.ERROR
    elif args.verbose == 1:
        loglevel = logging.WARNING
    elif args.verbose == 2:
        loglevel = logging.INFO
    else:  # args.verbose >= 3:
        loglevel = logging.DEBUG
    logging.basicConfig(level=loglevel)

    # Initialize random (used for stratification).
    seed(2709)

    # Report timing in INFO and DEBUG mode.
    start_ts = time()

    # Create prefix for output files, or use specified value.
    if not args.prefix:
        out_prefix = os.path.join(os.path.dirname(args.input),
                                  f'{int(time())}_')
    else:
        out_prefix = args.prefix
    logging.info(f'Output will be written with prefix {out_prefix}.')

    # Try to open in- and out files. If there are issues, better to fail now.
    try:
        infile_name = args.input
        infile = open(infile_name, 'r')

        flatfile_name = f'{out_prefix}flat.ldjson'
        balanced_flatfile_name = f'{out_prefix}balanced_flat.ldjson'
        rawfile_name = f'{out_prefix}_raw.ldjson'
        balanced_rawfile_name = f'{out_prefix}balanced_raw.ldjson'
        allmapsfile_name = f'{out_prefix}maps.json'
        featuremapfile_name = f'{out_prefix}feature_map.tsv'
        hostmapfile_name = f'{out_prefix}host_map.tsv'
        registeredmapfile_name = f'{out_prefix}registered_map.tsv'
        suffixmapfile_name = f'{out_prefix}suffix_map.tsv'

        for fn in [flatfile_name, balanced_flatfile_name,
                   rawfile_name, balanced_rawfile_name,
                   allmapsfile_name, featuremapfile_name, hostmapfile_name,
                   registeredmapfile_name, suffixmapfile_name]:
            if os.path.isfile(fn):
                if not confirm_overwrite(fn):
                    logging.critical(f'File {fn} exists and should not be '
                                     'overwritten. Aborting')
                    exit(-2)

        if args.unbalanced:
            flatfile = open(flatfile_name, 'w')
            rawfile = open(rawfile_name, 'w')
        balanced_flatfile = open(balanced_flatfile_name, 'w')
        balanced_rawfile = open(balanced_rawfile_name, 'w')
        allmapsfile = open(allmapsfile_name, 'w')
        featuremapfile = open(featuremapfile_name, 'w')
        hostmapfile = open(hostmapfile_name, 'w')
        registeredmapfile = open(registeredmapfile_name, 'w')
        suffixmapfile = open(suffixmapfile_name, 'w')

    except Exception as e:
        logging.critical(f'Failed to open files, aborting: {repr(e)}')
        exit(-1)

    # Try to read maps and meta info from reference file.
    try:
        logging.debug('Trying to open and parse reference file.')
        with open(args.reference, 'r') as ref_file:
            ref = simplejson.load(ref_file)
            reference_ts = ref['reference_ts']
            neutral_count = ref['neutral_count']
            non_neutral_count = ref['non_neutral_count']
            feature_map = ref['feature_map']
            max_feature_id = max(feature_map.values())
            host_map = ref['host_map']
            max_host_id = max(host_map.values())
            registered_map = ref['registered_map']
            max_registered_id = max(registered_map.values())
            suffix_map = ref['suffix_map']
            max_suffix_id = max(suffix_map.values())
            logging.info('Successfully read and parsed references from '
                         f'{args.reference}.')
    except Exception as e:
        logging.debug(f'Failed to load reference file: {repr(e)}')
        # If no (valid) reference file was specified, iterate over input data
        #   to determine the minimum queryTS, to get a count of neutral- and
        #   non-neutral records for sampling, and to establish feature-
        #   and domain mappings.
        reference_ts = 9999999999  # Very distant future
        neutral_count = 0
        non_neutral_count = 0
        feature_map = {}
        max_feature_id = 0
        host_map = {}
        max_host_id = 0
        registered_map = {}
        max_registered_id = 0
        suffix_map = {}
        max_suffix_id = 0

    logging.info('Iterating over input file - reference pass.')
    for (line_no, line) in enumerate(infile):
        # Report on progress every 1M (info) or 100k (debug) lines.
        if loglevel is logging.INFO:
            if not line_no % 1000000:
                logging.info(f'Reached line {line_no} after '
                             f'{int(time() - start_ts)} seconds.')
        elif loglevel is logging.DEBUG:
            if not line_no % 10000:
                logging.debug(f'Reached line {line_no} after '
                              f'{int(time() - start_ts)} seconds.')

        try:
            jline = simplejson.loads(line)

            # Keep track of timestamp (searching for the oldest query).
            reference_ts = min(reference_ts, int(jline['queryTS']))

            # Keep track of ratio of "neutral" target_verdict.
            if jline['base_score'] < .5:
                neutral_count += 1
            else:
                non_neutral_count += 1

            # Iterate over "response" array to build
            #   feature- and domain maps.
            for resp in jline['response']:

                if 'ruleHits' in resp:
                    ruleHits = resp['ruleHits']['tokens']
                    for orig_val in ruleHits:
                        if orig_val not in feature_map:
                            feature_map[orig_val] = max_feature_id
                            max_feature_id += 1

                if 'keyword' in resp['domain']:
                    orig_val = resp['domain']['keyword']
                    if orig_val not in host_map:
                        host_map[orig_val] = max_host_id
                        max_host_id += 1

                if 'registered' in resp['domain']:
                    orig_val = resp['domain']['registered']
                    if orig_val not in registered_map:
                        registered_map[orig_val] = max_registered_id
                        max_registered_id += 1

                if 'suffix' in resp['domain']:
                    orig_val = resp['domain']['suffix']
                    if orig_val not in suffix_map:
                        suffix_map[orig_val] = max_suffix_id
                        max_suffix_id += 1

        except Exception as e:
            logging.debug(f'Error parsing line {line_no}: {repr(e)}\n{line}')
            continue

    logging.info(f'Done iterating over {infile_name} - reference pass - '
                 f'took {int(time() - start_ts)} seconds.')

    # Return to beginning of infile.
    infile.seek(0)

    # Write domain and feature maps to files.
    map_ts = time()
    logging.info(f'Writing combined maps and meta info as json.')
    simplejson.dump(dict(feature_map=feature_map, host_map=host_map,
                         registered_map=registered_map, suffix_map=suffix_map,
                         reference_ts=reference_ts,
                         non_neutral_count=non_neutral_count,
                         neutral_count=neutral_count), allmapsfile)
    allmapsfile.close()

    logging.info(f'Writing map for {max_feature_id} features.')
    for (orig, mask) in feature_map.items():
        print(f'{orig}\t{mask}', file=featuremapfile)
    featuremapfile.close()

    logging.info(f'Writing map for {max_host_id} hosts.')
    for (orig, mask) in host_map.items():
        print(f'{orig}\t{mask}', file=hostmapfile)
    hostmapfile.close()

    logging.info(f'Writing map for {max_registered_id} '
                 'registered domains.')
    for (orig, mask) in registered_map.items():
        print(f'{orig}\t{mask}', file=registeredmapfile)
    registeredmapfile.close()

    logging.info(f'Writing map for {max_suffix_id} suffixes.')
    for (orig, mask) in suffix_map.items():
        print(f'{orig}\t{mask}', file=suffixmapfile)
    suffixmapfile.close()

    logging.info('Done writing map files - '
                 f'took {int(time() - map_ts)} seconds.')

    # Iterate over lines in input (again):
    # * Replace sensitive values with generic IDs accordingly.
    # * Compute queryTS with respect to the reference timestamp.
    # * Add target_score and target_verdict fields.
    # * Rename several fields.
    # * Downsample neutral target_verdicts.
    out_ts = time()
    logging.info('Iterating over input file and generating output.')
    for (line_no, line) in enumerate(infile):
        # Report on progress every 1M (info) or 100k (debug) lines.
        if loglevel is logging.INFO:
            if not line_no % 1000000:
                logging.info(f'Reached line {line_no} after '
                             f'{int(time() - out_ts)} seconds.')
        elif loglevel is logging.DEBUG:
            if not line_no % 10000:
                logging.debug(f'Reached line {line_no} after '
                              f'{int(time() - out_ts)} seconds.')

        try:
            jline = simplejson.loads(line)
            logging.debug(f'original: {jline}')  # original

            # Skip line if no base score is found.
            try:
                base_score = float(jline.pop('base_score'))
            except:
                continue
            # Get support 1 if present.
            try:
                support_1 = jline.pop('support_1')
            except:
                support_1 = 'unknown'
            # Get current support_2 if present.
            try:
                support_2 = jline.pop('support_2')
            except:
                support_2 = 'unknown'

            # Combine base score and support into final verdict and score.
            jline['target_score'] = base_score
            # Lower bound to suspect/positive midpoints if support convicted.
            if support_1 == 'spam' or support_2 == 'spam':
                jline['target_score'] = max(jline['target_score'], .95)
            elif support_1 == 'suspect' or support_2 == 'suspect':
                jline['target_score'] = max(jline['target_score'], .70)

            # Use target_score directly for regression-models.
            # Mapping into verdicts for classification-models:
            if jline['target_score'] >= .90:
                jline['target_verdict'] = '+'
            elif jline['target_score'] >= .50:
                jline['target_verdict'] = '?'
            else:
                jline['target_verdict'] = '-'

            support_fields = []  # redacted...
            # Remove remaining (sensitive) support fields.
            for sf in support_fields:
                if sf in jline:
                    _ = jline.pop(sf)

            # Replace original queryTS with offset to reference timestamp.
            orig_val = jline['queryTS']
            jline['queryTS'] = int(orig_val - reference_ts)

            # Iterate over the "response" array.
            for resp in jline['response']:
                # Replace domain-related values with IDs.
                if 'keyword' in resp['domain']:
                    orig_val = resp['domain'].pop('keyword')
                    resp['domain']['host'] = host_map[orig_val]

                if 'registered' in resp['domain']:
                    orig_val = resp['domain']['registered']
                    resp['domain']['registered'] = registered_map[orig_val]

                if 'suffix' in resp['domain']:
                    orig_val = resp['domain']['suffix']
                    resp['domain']['suffix'] = suffix_map[orig_val]

                # Replace feature labels with IDs.
                resp['features'] = []
                if 'ruleHits' in resp:
                    ruleHits = resp.pop('ruleHits')['tokens']
                    for orig_val in ruleHits:
                        resp['features'].append(feature_map[orig_val])

        except Exception as e:
            logging.warning(f'Failed to parse line {line_no}: {repr(e)}')
            continue

        # Keep raw line (nested json).
        rawline = simplejson.dumps(jline, separators=(',', ':'))
        if args.unbalanced:
            print(rawline, file=rawfile)

        # "Flatten" (or one-hot-encode) json so it better fits in dataframes.
        flatten_record(jline, list(feature_map.values()))
        flatline = simplejson.dumps(jline, separators=(',', ':'))
        logging.debug(f'flat: {flatline}')  # flattened
        if args.unbalanced:
            print(flatline, file=flatfile)

        # Downsample "neutral" target verdicts and write to balanced file.
        if (jline['target_verdict'] == '-' and
                random() > non_neutral_count / neutral_count):
            continue
        print(flatline, file=balanced_flatfile)
        print(rawline, file=balanced_rawfile)

    if args.unbalanced:
        flatfile.close()
        rawfile.close()
    balanced_flatfile.close()
    balanced_rawfile.close()
    infile.close()
    logging.info(f'Done iterating over {infile_name} - main pass - '
                 f'took {int(time() - out_ts)} seconds.')

    logging.info(f'Total time: {int(time() - start_ts)} seconds.')
