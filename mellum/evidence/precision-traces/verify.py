#!/usr/bin/env python3
"""Verify the committed trace collection using only Python's standard library."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parse-json', action='store_true',
                        help='Also parse every trace and check event/category counts; requires more RAM.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['traces'] + manifest['supporting_files'] + manifest.get('documents', []):
        path = (root / entry['path']).resolve()
        assert path.is_relative_to(root), entry['path']
        assert path.stat().st_size == entry['bytes'], entry['path']
        assert checksum(path) == entry['sha256'], entry['path']
    for entry in manifest['traces']:
        digest = hashlib.sha256()
        size = 0
        with gzip.open(root / entry['path'], 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
                size += len(chunk)
        assert digest.hexdigest() == entry['raw_sha256'], entry['path']
        assert size == entry['raw_bytes'], entry['path']
        if args.parse_json:
            with gzip.open(root / entry['path'], 'rt', encoding='utf-8') as stream:
                document = json.load(stream)
            if entry['kind'] == 'chrome_trace':
                events = document if isinstance(document, list) else document['traceEvents']
                assert len(events) == entry['event_count'], entry['path']
                categories = collections.Counter(event.get('cat') or '(none)' for event in events)
                assert dict(categories) == entry['categories'], entry['path']
            del document
    print(json.dumps({'verified_traces': len(manifest['traces']),
                      'verified_supporting_files': len(manifest['supporting_files']),
                      'parsed_json': args.parse_json}, indent=2))


if __name__ == '__main__':
    main()
