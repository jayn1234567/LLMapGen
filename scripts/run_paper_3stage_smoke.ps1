$ErrorActionPreference = 'Stop'

Write-Host '[Stage1] Training...'
python -m unimapgen.train --config configs/unimapgen_paper_stage1_sft_smoke.yaml

Write-Host '[Stage2] Training...'
python -m unimapgen.train --config configs/unimapgen_paper_stage2_align_smoke.yaml

Write-Host '[Stage3] Training...'
python -m unimapgen.train --config configs/unimapgen_paper_stage3_state_smoke.yaml

Write-Host '[Stage3] Eval...'
python -m unimapgen.eval --config configs/unimapgen_paper_stage3_state_smoke.yaml --checkpoint outputs/unimapgen_paper_stage3_state_smoke/latest.pt

Write-Host '[Stage3] Global state scan...'
python -m unimapgen.infer_state_scan --config configs/unimapgen_paper_stage3_state_smoke.yaml --checkpoint outputs/unimapgen_paper_stage3_state_smoke/latest.pt --split val --scene_limit 1 --max_patches_per_scene 8 --output outputs/paper_stage3_state_scan.json

Write-Host 'Done.'
