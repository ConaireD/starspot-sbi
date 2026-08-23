"""
Collect the numbers the paper notebooks printed, into one markdown file.

Reads .ipynb files and nothing else. It never executes a notebook, never
writes to one, and never touches the figure outputs: the numbers come from
the stored `outputs` of the last run, so what this produces is a transcript
of that run and not a fresh evaluation. A notebook whose code has changed
since it was executed will report the old numbers with no warning from the
kernel, which is what the staleness flags below are for.

The default output is NUMBERS_generated.md, deliberately not NUMBERS.md:
that file is curated prose with consequences and caveats written by hand,
and regenerating over it would lose them. Point --out at it only if that is
what you actually mean.

Usage:

    python scripts/collect_numbers.py
    python scripts/collect_numbers.py --notebooks ../staging/notebooks/paper
    python scripts/collect_numbers.py --out /tmp/numbers.md --quiet

Each notebook becomes one section: its title and draft section, a staleness
line when the stored run was not a clean top-to-bottom pass, the figure
files it reported writing, and its printed output with progress chatter
dropped.
"""

import argparse
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_NOTEBOOKS = os.path.join(REPO, 'notebooks', 'paper')
DEFAULT_OUT = os.path.join(REPO, 'NUMBERS_generated.md')

# Lines that carry no number worth keeping: chunk counters ("2,000 / 100,000"),
# progress bars, and the bare rung/step tags a long pass prints as it goes.
NOISE = (
    re.compile(r'^\s*[\d,]+\s*/\s*[\d,]+\s*$'),
    re.compile(r'^\s*\d+%\|'),
    re.compile(r'^\s*$'),
)
# save_fig's report. Pulled out and listed separately rather than dropped: it
# is the only record in the notebook of which files the run actually wrote.
WROTE = re.compile(r'^wrote (.+)$')
# The draft section a notebook serves, from its opening markdown cell.
SECTION = re.compile(r'Section ([\w.]+),\s*`([^`]+)`')


def stdout_of(cell):
    """Every stdout line a code cell stored, in order. stderr is ignored."""
    lines = []
    for out in cell.get('outputs', []):
        if out.get('output_type') == 'stream' and out.get('name') == 'stdout':
            lines.extend(''.join(out.get('text', [])).splitlines())
    return lines


def errors_of(cell):
    """The exception names a code cell stored, if it ended in one."""
    return [out.get('ename', 'error') for out in cell.get('outputs', [])
            if out.get('output_type') == 'error']


def staleness(code_cells):
    """
    What is off about the stored run, as a list of short phrases.

    A clean pass executes every non-empty cell exactly once, in order, so the
    execution counts read 1, 2, 3 ... with nothing missing. Anything else means
    the stored outputs come from more than one run and cannot be read as a
    single transcript.
    """
    notes = []
    live = [c for c in code_cells if ''.join(c['source']).strip()]
    counts = [c.get('execution_count') for c in live]
    if all(n is None for n in counts):
        return ['never executed: no stored outputs']
    if any(n is None for n in counts):
        notes.append(f'{sum(n is None for n in counts)} of {len(live)} cells '
                     f'not executed in the stored run')
    ran = [n for n in counts if n is not None]
    if ran != sorted(ran):
        notes.append('cells executed out of order')
    if ran and ran != list(range(1, len(ran) + 1)):
        notes.append(f'execution counts run {min(ran)} to {max(ran)} over '
                     f'{len(ran)} cells, so this is not one clean pass')
    return notes


def heading(nb, name):
    """(title, section phrase) from the notebook's opening markdown cell."""
    title, section = name, None
    for cell in nb['cells']:
        if cell['cell_type'] != 'markdown':
            continue
        text = ''.join(cell['source'])
        for line in text.splitlines():
            if line.startswith('# '):
                title = line[2:].strip()
                break
        hit = SECTION.search(text)
        if hit:
            section = f'Section {hit.group(1)}, {hit.group(2)}'
        break
    return title, section


def collect(path):
    """One notebook's section, as a list of markdown lines."""
    with open(path) as f:
        nb = json.load(f)

    name = os.path.basename(path)
    title, section = heading(nb, name)
    code = [c for c in nb['cells'] if c['cell_type'] == 'code']

    numbers, wrote, failed = [], [], []
    for cell in code:
        failed.extend(errors_of(cell))
        for line in stdout_of(cell):
            hit = WROTE.match(line)
            if hit:
                wrote.extend(p.strip() for p in hit.group(1).split(','))
            elif not any(p.match(line) for p in NOISE):
                numbers.append(line)

    out = [f'## {name} — {section}' if section else f'## {name}', '']
    if title and title != name:
        out += [f'{title}.', '']
    for note in staleness(code):
        out += [f'> STALE: {note}', '']
    if failed:
        out += [f'> FAILED: the stored run raised {", ".join(sorted(set(failed)))}',
                '']
    if numbers:
        out += ['```'] + numbers + ['```', '']
    else:
        out += ['No printed output stored.', '']
    if wrote:
        out += ['Wrote ' + ', '.join(f'`{w}`' for w in wrote) + '.', '']
    return out, bool(numbers)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--notebooks', default=DEFAULT_NOTEBOOKS,
                    help='directory of .ipynb files (default: %(default)s)')
    ap.add_argument('--out', default=DEFAULT_OUT,
                    help='markdown file to write (default: %(default)s)')
    ap.add_argument('--quiet', action='store_true',
                    help='do not report per-notebook progress on stderr')
    a = ap.parse_args(argv)

    if not os.path.isdir(a.notebooks):
        ap.error(f'{a.notebooks} is not a directory')
    paths = sorted(os.path.join(a.notebooks, f)
                   for f in os.listdir(a.notebooks) if f.endswith('.ipynb'))
    if not paths:
        ap.error(f'no notebooks in {a.notebooks}')

    body, empty = [], []
    for p in paths:
        try:
            lines, had = collect(p)
        except (ValueError, KeyError) as exc:
            lines, had = ([f'## {os.path.basename(p)}', '',
                           f'> UNREADABLE: {exc}', ''], False)
        body += lines
        if not had:
            empty.append(os.path.basename(p))
        if not a.quiet:
            print(f'  {os.path.basename(p)}'
                  f'{"" if had else "   (no output)"}', file=sys.stderr)

    head = [
        '# Numbers the notebooks printed',
        '',
        f'Generated by `scripts/collect_numbers.py` from the stored outputs '
        f'of {len(paths)} notebooks in `{os.path.relpath(a.notebooks, REPO)}`.',
        '',
        'These are a transcript of each notebook\'s last run, not a fresh '
        'evaluation: nothing here was executed to produce this file. A block '
        'marked STALE came from more than one run and should not be read as '
        'one set of numbers. Verify against the producing notebook before '
        'quoting anything in the draft.',
        '',
    ]
    if empty:
        head += ['Notebooks with no stored output: '
                 + ', '.join(f'`{e}`' for e in empty) + '.', '']

    with open(a.out, 'w') as f:
        f.write('\n'.join(head + body).rstrip() + '\n')
    print(f'wrote {a.out} from {len(paths)} notebooks '
          f'({len(empty)} with no output)', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
