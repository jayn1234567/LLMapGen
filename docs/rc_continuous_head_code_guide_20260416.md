# RC 杩炵画鍥炲綊澶翠唬鐮佽鏄?
鏃ユ湡锛歚2026-04-16`

## 1. 鏂囨。鐩殑

杩欎唤鏂囨。涓撻棬璇存槑鎴戜滑 RC 杩炵画鍥炲綊澶磋繖涓€鏀唬鐮佺殑锛?
- 浠ｇ爜浣嶇疆
- 涓昏瀹炵幇鎬濊矾
- 鏁版嵁鏄€庝箞缁勭粐鐨?- 鎹熷け鏄€庝箞璁＄畻鐨?- 涓庢櫘閫?JSON SFT 璺嚎鐨勫尯鍒?
杩欓噷璇寸殑鈥滆繛缁洖褰掑ご鈥濓紝鎸囩殑鏄細

- 妯″瀷浠嶇劧浣跨敤 `DINOv2 -> visual bridge -> Qwen`
- 浣嗕笉鍐嶈 Qwen 鐩存帴鑷洖褰掕緭鍑虹鏁ｅ潗鏍囨暟瀛?- 鑰屾槸鍦ㄨ瑷€妯″瀷闅愮姸鎬佷笂鍐嶆帴涓€涓皬鍨?`coord_head`
- 瀵瑰潗鏍囧崰浣?token 瀵瑰簲鐨勪綅缃洿鎺ュ洖褰掕繛缁殑 `(x, y)`

## 2. 涓昏浠ｇ爜浣嶇疆

### 2.1 鏁版嵁闆嗕笌鍗犱綅鏍煎紡

鏂囦欢锛?
- `unimapgen/data/rc_centerline_continuous_head_dataset.py`

鑱岃矗锛?
- 鎶婂師濮?`centerline_json` 璇诲嚭鏉?- 鎶婃瘡涓湡瀹炵偣鏇挎崲鎴愬崰浣?token `<coord_pt>`
- 鍚屾椂鎶婄湡瀹炲潗鏍囦繚瀛樻垚杩炵画鐩戠潱鐩爣 `coord_targets`

杩欎釜鏂囦欢閲屾渶閲嶈鐨勫嚑涓偣锛?
- `CONTINUOUS_COORD_TOKEN = "<coord_pt>"`
- `build_placeholder_centerline_json(...)`
- `RCCenterlineContinuousHeadDataset`
- `RCCenterlineContinuousHeadCollator`

### 2.2 妯″瀷涓讳綋

鏂囦欢锛?
- `unimapgen/models/qwen3_rc_dinov2_centerline_continuous_head.py`

鑱岃矗锛?
- 澶嶇敤鍘熸湁 `Qwen3RCDinoCenterlineJSONSFTModel`
- 淇濈暀鍘熸潵鐨勮瑙夋ˉ鎺ヤ笌璇█妯″瀷缁撴瀯
- 鍦ㄦ渶鍚庡鍔犱竴涓皬鍨?`coord_head`
- 浠庤瑷€妯″瀷鏈€鍚庝竴灞傞殣鐘舵€佸洖褰掕繛缁潗鏍?
鏈€鍏抽敭鐨勫疄鐜扮偣锛?
- `Qwen3RCDinoCenterlineContinuousHeadModel`
- `self.coord_head = nn.Sequential(...)`
- `predict_coordinates(...)`
- `forward(...)`

### 2.3 璁粌涓庢崯澶?
鏂囦欢锛?
- `scripts/train_qwen3_rc_dinov2_centerline_continuous_head.py`

鑱岃矗锛?
- 缁勮 tokenizer / dataset / model / trainer
- 瀹氫箟璇█鎹熷け鍜屽潗鏍囧洖褰掓崯澶辩殑缁勫悎鏂瑰紡
- 淇濆瓨杩炵画鍥炲綊澶寸浉鍏虫ā鍧?
鏈€鍏抽敭鐨勫疄鐜扮偣锛?
- `RCDinoContinuousHeadTrainer`
- `_compute_coord_loss(...)`
- `compute_loss(...)`

### 2.4 棣栧熬鐐逛紭鍏堣繛缁洖褰掑彉浣?
濡傛灉鐪嬬殑鏄€滃厛棣栫偣銆佸啀灏剧偣銆佹渶鍚庡唴閮ㄧ偣鈥濈殑杩炵画鍥炲綊鐗堟湰锛屽搴旀枃浠舵槸锛?
- `unimapgen/data/rc_centerline_startend_continuous_head_dataset.py`
- `scripts/train_qwen3_rc_dinov2_centerline_startend_continuous_head.py`

杩欎竴鐗堟湰璐ㄤ笂杩樻槸杩炵画鍥炲綊澶达紝鍙槸鎶婁竴鏉＄嚎鐨勭洃鐫ｉ『搴忔敼鎴愪簡锛?
1. `start`
2. `end`
3. `inner_points`

## 3. 鏁翠綋缁撴瀯

杩炵画鍥炲綊澶磋矾绾垮彲浠ユ鎷负锛?
```text
RC patch
-> DINOv2
-> visual_norm
-> visual_projector
-> geometric_position_mlp
-> token_alignment
-> 娉ㄥ叆鍒?Qwen 鐨?<vis_patch>
-> Qwen 鏈€鍚庝竴灞?hidden states
-> coord_head
-> 杩炵画鍧愭爣 (x, y)
```

杩欓噷鏈€閲嶈鐨勪竴鐐规槸锛?
- 瑙嗚妗ユ帴閮ㄥ垎娌℃湁鍙﹁捣鐐夌伓
- 浠嶇劧娌跨敤鎴戜滑鐜版湁鐨?DINOv2 -> Qwen 娉ㄥ叆鏂规
- 鏂板鐨勫彧鏄€滃浣曚粠鏌愪簺 token 鐨?hidden state 璇诲嚭杩炵画鍧愭爣鈥?
## 4. 鏁版嵁鏄€庝箞缁勭粐鐨?
### 4.1 assistant 鐩爣涓嶆槸鐩存帴鍐欑湡瀹炵偣

鍦ㄨ繛缁洖褰掑ご鐗堟湰閲岋紝assistant 鐩爣涓嶆槸锛?
```json
{"lines":[{"points":[[x1,y1],[x2,y2]]}]}
```

鑰屾槸浼氬厛杞垚鍗犱綅褰㈠紡锛屼緥濡傦細

```json
{"lines":[{"points":["<coord_pt>","<coord_pt>"]}]}
```

涔熷氨鏄锛?
- 鏂囨湰搴忓垪閲屽彧淇濈暀缁撴瀯楠ㄦ灦
- 鐪熷疄鍧愭爣涓嶅啀閫氳繃璇█ token 鐩存帴琛ㄨ揪
- 鐪熷疄鍧愭爣鍗曠嫭鏀惧湪 `coord_targets` 閲?
### 4.2 collator 浼氶澶栫敓鎴愯繛缁洃鐫ｅ紶閲?
`RCCenterlineContinuousHeadCollator` 闄や簡甯歌鐨勶細

- `input_ids`
- `attention_mask`
- `labels`
- `pixel_values`
- `vis_patch_mask`

杩樹細棰濆鐢熸垚锛?
- `coord_target_values`

瀹冪殑褰㈢姸鏄細

```text
[batch, seq_len, 2]
```

鍏朵腑锛?
- 闈炲潗鏍囦綅缃～ `-1`
- 鍧愭爣鍗犱綅 token 瀵瑰簲浣嶇疆濉湡瀹炲綊涓€鍖?`(x, y)`

## 5. 妯″瀷鏄€庝箞鍋氳繛缁潗鏍囬娴嬬殑

### 5.1 鍏堢収甯稿仛瑙嗚娉ㄥ叆

妯″瀷鍏堣蛋涓?JSON SFT 鐩稿悓鐨勪富骞诧細

- 鐢?DINOv2 鎻愬彇瑙嗚鐗瑰緛
- 缁忚繃 `visual_norm`
- 鍐嶈繃 `visual_projector`
- 鍐嶈繃 `geometric_position_mlp`
- 鍐嶈繃 `token_alignment`
- 鏈€缁堟浛鎹㈣緭鍏ュ簭鍒椾腑鐨?`<vis_patch>`

### 5.2 鐒跺悗璇诲彇璇█妯″瀷鏈€鍚庝竴灞傞殣鐘舵€?
鍦?`forward(...)` 閲岋紝Qwen 杈撳嚭锛?
- `loss`
- `logits`
- `hidden_states`

杩炵画鍥炲綊澶村叧蹇冪殑鏄渶鍚庝竴灞傦細

```text
outputs.hidden_states[-1]
```

鐒跺悗鎶婂畠閫佽繘锛?
```python
self.coord_head
```

寰楀埌锛?
```text
coord_pred[..., 2]
```

涔熷氨鏄瘡涓?token 浣嶇疆瀵瑰簲涓€涓繛缁?`(x, y)` 棰勬祴銆?
### 5.3 涓轰粈涔堜笉鐢ㄥ鎵€鏈変綅缃兘鍥炲綊

铏界劧 `coord_head` 浼氬鏁存潯搴忓垪鎵€鏈変綅缃兘杈撳嚭 `(x, y)`锛屼絾鐪熸鍙備笌鐩戠潱鐨勫彧鏈夛細

- 涓嬩竴涓?token 鏄潗鏍囧崰浣?token 鐨勯偅浜涗綅缃?
涔熷氨鏄锛屽彧鏈夊拰 `<coord_pt>` 瀵归綈鐨勪綅缃墠浼氳绠楀洖褰掓崯澶便€?
## 6. 鎹熷け鏄€庝箞璁＄畻鐨?
杩炵画鍥炲綊澶磋缁冩椂锛屾渶缁?loss 鐢变袱閮ㄥ垎缁勬垚锛?
### 6.1 璇█鎹熷け

杩欓儴鍒嗚繕鏄?Qwen 鏍囧噯鐨勮嚜鍥炲綊浜ゅ弶鐔碉細

```text
base_loss = outputs.loss
```

瀹冭礋璐ｅ涔狅細

- JSON 楠ㄦ灦
- 琛岀粨鏋?- `lines / points` 杩欎簺鏂囨湰缁勭粐鏂瑰紡

### 6.2 鍧愭爣鍥炲綊鎹熷け

璁粌鑴氭湰閲岀敤鐨勬槸锛?
- `SmoothL1Loss`

瀹炵幇浣嶇疆锛?
- `RCDinoContinuousHeadTrainer._compute_coord_loss(...)`

鏈変竴涓緢閲嶈鐨勭粏鑺傦細

```python
shift_pred = coord_pred[:, :-1, :]
shift_target = coord_target_values[:, 1:, :]
```

鍘熷洜鏄細

- 璇█妯″瀷鏄?next-token 棰勬祴
- 鎵€浠ュ綋鍓嶄綅缃殑 hidden state锛屽榻愮殑鏄€滀笅涓€涓?token鈥?- 杩炵画鍧愭爣鐩戠潱涔熻鍋氬悓鏍风殑 shift

鍙湁鐩爣浣嶇疆鏈夋晥鏃舵墠鍙備笌鍥炲綊锛?
```python
valid_mask = shift_target[..., 0].ge(0.0) & shift_target[..., 1].ge(0.0)
```

鏈€缁堟€绘崯澶憋細

```text
total_loss = base_loss + coord_loss_weight * coord_reg_loss
```

榛樿杩樹細璁板綍锛?
- `coord_reg_loss`
- `coord_reg_mae`

## 7. 淇濆瓨浜嗗摢浜涙ā鍧?
璁粌瀹屾垚鍚庯紝杩炵画鍥炲綊澶翠細鎶婄浉鍏虫ā鍧椾竴璧蜂繚瀛樺埌锛?
- `rc_dinov2_centerline_continuous_head_modules.pt`

鍏朵腑鍖呮嫭锛?
- `vision_encoder`
- `visual_norm`
- `visual_projector`
- `geometric_position_mlp`
- `token_alignment`
- `coord_head`
- `special_token_adapter`

杩欐剰鍛崇潃鍚庣画鎭㈠璁粌鎴栧崟鐙嬁杩欐潯绾垮仛鎺ㄧ悊鏃讹紝涓嶅彧淇濈暀浜嗚瑷€妯″瀷 LoRA锛屼篃淇濈暀浜嗚繛缁洖褰掑ご鑷繁鐨勫弬鏁般€?
## 8. 涓庢櫘閫?JSON SFT 鐨勫尯鍒?
鏅€?JSON SFT锛?
- 鐩存帴璁?Qwen 鐢熸垚鍧愭爣鏁板瓧 token
- 鍧愭爣瀹屽叏璧扮鏁ｈ瘝琛?
杩炵画鍥炲綊澶达細

- Qwen 鍙礋璐ｇ敓鎴愮粨鏋勯鏋跺拰鍧愭爣鍗犱綅
- 鐪熸鐨勫潗鏍囧€肩敱 `coord_head` 鐩存帴杩炵画鍥炲綊

鍙互鎶婂畠鐞嗚В鎴愶細

- 鏅€?JSON SFT 鏇村儚鈥滄妸鍧愭爣褰撴枃鏈啓鍑烘潵鈥?- 杩炵画鍥炲綊澶存洿鍍忊€滄妸鍧愭爣褰撴暟鍊肩洿鎺ヨ鍑烘潵鈥?
## 9. 褰撳墠寤鸿鎬庝箞璇昏繖濂椾唬鐮?
濡傛灉瑕佸揩閫熺悊瑙ｏ紝寤鸿鎸夎繖涓『搴忕湅锛?
1. `unimapgen/data/rc_centerline_continuous_head_dataset.py`
2. `unimapgen/models/qwen3_rc_dinov2_centerline_continuous_head.py`
3. `scripts/train_qwen3_rc_dinov2_centerline_continuous_head.py`

杩欐牱浼氭渶瀹规槗鐪嬫竻锛?
- 鏁版嵁濡備綍浠庣湡瀹炵偣鍙樻垚鍗犱綅缁撴瀯
- hidden state 鎬庝箞琚槧灏勬垚杩炵画鍧愭爣
- 璇█鎹熷け鍜屽洖褰掓崯澶辨槸鎬庝箞鍚堝湪涓€璧疯缁冪殑
