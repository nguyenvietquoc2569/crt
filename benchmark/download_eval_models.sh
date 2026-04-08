#!/bin/bash
# Download Whisper and UTMOS models for benchmark evaluation
# Run on login node (has internet): ssh eaic "bash /datastore/eaic/nvquoc/mimi_vc/benchmark/download_eval_models.sh"

set -e
source /datastore/eaic/nvquoc/mimi_vc_env/bin/activate

export TORCH_HOME=/datastore/eaic/nvquoc/mimi_vc/data_storage/torch_cache

echo "=== Installing eval dependencies ==="
pip install --quiet openai-whisper 2>/dev/null || pip install --quiet git+https://github.com/openai/whisper.git

echo "=== Downloading Whisper base model ==="
python3 -c "
import whisper
model = whisper.load_model('base', device='cpu')
print('Whisper base model downloaded and cached.')
"

echo "=== Downloading UTMOS model ==="
python3 -c "
import torch
model = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)
print('UTMOS model downloaded and cached.')
"

echo "=== Done ==="
