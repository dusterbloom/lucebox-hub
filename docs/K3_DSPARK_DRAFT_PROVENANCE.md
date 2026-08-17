# Kimi K3 DSpark draft provenance

STATUS: CONVERTED / RUNTIME VALIDATION OPEN

The shortest compatible draft path is the five-layer
`RadixArk/Kimi-K3-DSpark` safetensors checkpoint converted with Lucebox's
existing `server/scripts/convert_dspark_to_gguf.py`.  Its topology matches the
generic Kimi draft graph: hidden width 7168, five GQA layers, capture layers
7/23/51/67/83, vocabulary 163840, and DSpark Markov rank 256.

## Frozen source

- Repository: `RadixArk/Kimi-K3-DSpark`
- Revision: `3c5bac301d9cf392706189d82ed947feca6c2f0f`
- Source file: `model.safetensors`
- Source bytes: `4,498,585,858`
- Source SHA-256:
  `ecd746459b4a603ce0d2c64f73935efead29bd651b14439b99e57ee8b41b77ca`

The repository currently publishes no clear license file or model-card license
field.  The converted artifact is therefore for internal evaluation only and
must not be redistributed until the weight license is clarified.

## Converted artifact

- Local file: `/mnt/kimi-k3/models/kimi-k3-dspark-radixark-q8_0.gguf`
- Bytes: `2,390,153,920`
- SHA-256:
  `6120005799a768239fe47edd05fc500672eb6eea2d22c089fcfa13e9de56de9a`
- Quantized tensors: 38 Q8_0 plus 24 F32 tensors
- Q8 parameter count: `2,249,195,520`
- Sampled relative RMSE, RMS: `0.0054412725`
- Sampled relative RMSE, maximum: `0.0055733171`
- Converter ceiling: `0.01`

The source hash and conversion error gate passed.  Static compatibility and
conversion are measured complete.  Runtime loading, draft acceptance length,
and end-to-end speculative speed remain open because the WSL CUDA device was
not available after conversion.

## Reproduction

```bash
hf download RadixArk/Kimi-K3-DSpark \
  config.json model.safetensors README.md \
  --revision 3c5bac301d9cf392706189d82ed947feca6c2f0f \
  --local-dir /mnt/kimi-k3/models/kimi-k3-dspark-radixark

python3 server/scripts/convert_dspark_to_gguf.py \
  /mnt/kimi-k3/models/kimi-k3-dspark-radixark \
  /mnt/kimi-k3/models/kimi-k3-dspark-radixark-q8_0.gguf \
  --report /mnt/kimi-k3/models/kimi-k3-dspark-radixark-q8_0.json \
  --source-repo RadixArk/Kimi-K3-DSpark \
  --source-revision 3c5bac301d9cf392706189d82ed947feca6c2f0f \
  --target-repo moonshotai/Kimi-K3 \
  --expected-sha256 \
    ecd746459b4a603ce0d2c64f73935efead29bd651b14439b99e57ee8b41b77ca
```
