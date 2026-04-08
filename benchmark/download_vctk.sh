#!/bin/bash
# Download VCTK corpus for multi-pair VC evaluation
# Run on login node (has internet): ssh eaic "bash /datastore/eaic/nvquoc/mimi_vc/benchmark/download_vctk.sh"

set -e
source /datastore/eaic/nvquoc/mimi_vc_env/bin/activate

WORK_DIR=/datastore/eaic/nvquoc/mimi_vc
VCTK_DIR=$WORK_DIR/benchmark/vctk

if [ -d "$VCTK_DIR/wav48_silence_trimmed" ]; then
    echo "VCTK already downloaded at $VCTK_DIR"
    echo "Speakers: $(ls $VCTK_DIR/wav48_silence_trimmed/ | wc -l)"
    exit 0
fi

mkdir -p $VCTK_DIR
cd $VCTK_DIR

echo "=== Downloading VCTK corpus ==="
echo "Source: University of Edinburgh (CC BY 4.0)"
echo "This is ~11GB compressed, ~15GB extracted"

# VCTK 0.92 from official mirror
wget -q --show-progress https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip -O VCTK-Corpus-0.92.zip

echo "=== Extracting ==="
unzip -q VCTK-Corpus-0.92.zip -d .
rm VCTK-Corpus-0.92.zip

echo "=== VCTK downloaded ==="
echo "Speakers: $(ls wav48_silence_trimmed/ | wc -l)"
echo "Sample: $(ls wav48_silence_trimmed/p225/ | head -3)"

# Create evaluation pairs: 10 sources x 10 targets = 100 pairs
# Select 5 male + 5 female speakers for diversity
echo ""
echo "=== Setting up evaluation pairs ==="
python3 -c "
import os, random, json

vctk_wav = '$VCTK_DIR/wav48_silence_trimmed'
speakers = sorted([d for d in os.listdir(vctk_wav) if os.path.isdir(os.path.join(vctk_wav, d))])
print(f'Total speakers: {len(speakers)}')

# VCTK speaker metadata (subset)
# Males: p226, p227, p232, p237, p241, p243, p245, p246, p247, p251
# Females: p225, p228, p229, p230, p231, p233, p234, p236, p238, p239
source_speakers = ['p225', 'p228', 'p232', 'p237', 'p241',   # 3F + 2M sources
                   'p226', 'p229', 'p243', 'p245', 'p247']   # 2M + 3F (mixed)
target_speakers = ['p230', 'p231', 'p233', 'p234', 'p236',   # 5F targets
                   'p227', 'p238', 'p239', 'p246', 'p251']   # 2M + 3F targets

# Verify all speakers exist
for s in source_speakers + target_speakers:
    sp_dir = os.path.join(vctk_wav, s)
    if not os.path.isdir(sp_dir):
        print(f'  WARNING: {s} not found, skipping')
        continue

# For each source, pick 1 utterance (~3-8s)
# For each target, pick 5 utterances as reference corpus
pairs = []
for src in source_speakers:
    src_dir = os.path.join(vctk_wav, src)
    if not os.path.isdir(src_dir):
        continue
    src_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.flac')])
    if not src_files:
        continue
    # Pick utterance 001 as source
    src_file = src_files[0]

    for tgt in target_speakers:
        if tgt == src:
            continue
        tgt_dir = os.path.join(vctk_wav, tgt)
        if not os.path.isdir(tgt_dir):
            continue
        tgt_files = sorted([f for f in os.listdir(tgt_dir) if f.endswith('.flac')])
        if len(tgt_files) < 5:
            continue
        # Pick first 5 utterances as target reference
        tgt_refs = tgt_files[:5]

        pairs.append({
            'source_speaker': src,
            'target_speaker': tgt,
            'source_file': os.path.join(src_dir, src_file),
            'target_files': [os.path.join(tgt_dir, f) for f in tgt_refs],
        })

print(f'Generated {len(pairs)} evaluation pairs')
print(f'  Source speakers: {sorted(set(p[\"source_speaker\"] for p in pairs))}')
print(f'  Target speakers: {sorted(set(p[\"target_speaker\"] for p in pairs))}')

# Gender combos
combos = {}
# Approximate gender from VCTK metadata
females = {'p225','p228','p229','p230','p231','p233','p234','p236','p238','p239'}
males = {'p226','p227','p232','p237','p241','p243','p245','p246','p247','p251'}
for p in pairs:
    sg = 'F' if p['source_speaker'] in females else 'M'
    tg = 'F' if p['target_speaker'] in females else 'M'
    combo = f'{sg}->{tg}'
    combos[combo] = combos.get(combo, 0) + 1
print(f'  Gender combos: {combos}')

with open('$VCTK_DIR/eval_pairs.json', 'w') as f:
    json.dump(pairs, f, indent=2)
print(f'Saved to $VCTK_DIR/eval_pairs.json')
"

echo ""
echo "=== Done ==="
echo "VCTK: $VCTK_DIR"
echo "Eval pairs: $VCTK_DIR/eval_pairs.json"
