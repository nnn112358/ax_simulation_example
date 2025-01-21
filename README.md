
## Purpose

Simulate a depth anything using the ax630c simulator pulsar2 run.<br>
pulsar2 is a software environment released by axera-tech.<br>

https://github.com/AXERA-TECH/pulsar2-docs-en

## How to

Enter Docker in pulsar2

```
 sudo docker run -it --net host --rm -v $PWD:/data pulsar2:3.3
```



Check the interface of depth_anything_u8.axmodel

To run a simulation with a quantized axmodel
```
# python pulsar2_run_preprocess_axsim.py
# pulsar2 run --model depth_anything_u8.axmodel --input_dir sim_inputs --output_dir sim_outputs --list list.txt
# python pulsar2_run_postprosess_step1.py   --model depth_anything_u8.axmodel   --output-dir ./sim_outputs/0   --num-outputs 1   --bin1 ./sim_outputs/0/depth.bin
```

To perform a simulation with onnx before quantization

```
# python pulsar2_run_preprocess_onnx.py
# pulsar2 run --model depth_anything_u8.axmodel --input_dir sim_inputs --output_dir sim_outputs --list list.txt
# python pulsar2_run_postprosess_step1.py   --model depth_anything_u8.axmodel   --output-dir ./sim_outputs/0   --num-outputs 1   --bin1 ./sim_outputs/0/depth.bin
```

## Result

![image](https://github.com/user-attachments/assets/72efdf9c-7c70-44a3-9615-3248f091be30)



## Tools
```
# python axmodel_get_info.py depth_anything_u8.axmodel
INFO: モデルを読み込み中: depth_anything_u8.axmodel

=== ONNXモデル情報 ===
モデルパス: depth_anything_u8.axmodel
IRバージョン: 8
プロデューサー: Pulsar2 (バージョン: )
ドメイン:
モデルバージョン: 0

=== 入力テンソル情報 ===

入力名: image
形状: [1, 256, 384, 3]
データ型: uint8
総要素数: 294,912

=== 出力テンソル情報 ===

出力名: depth
形状: [1, 1, 256, 384]
データ型: float32
総要素数: 98,304
```


## pulsar2 document 

```
# pulsar2 run -h
usage: main.py run [-h] [--config] [--model] [--input_dir] [--output_dir] [--list] [--random_input ]
                   [--batch_size] [--enable_perlayer_output ] [--dump_with_stride ] [--group_index] [--mode]
                   [--target_hardware]

optional arguments:
  -h, --help            show this help message and exit
  --config              config file path, supported formats: json / yaml / toml / prototxt. type: string.
                        required: false. default:.
  --model               run model path, support ONNX, QuantAxModel and CompiledAxmodel. type: string. required:
                        true.
  --input_dir           model input data in this directory. type: string. required: true. default:.
  --output_dir          model output data directory. type: string. required: true. default:.
  --list                list file path. type: string. required: true. default:.
  --random_input []     random input data. type: bool. required: false. default: false.
  --batch_size          batch size to be used in dynamic inference mode, only work for CompiledAxModel. type: int.
                        required: false. defalult: 0.
  --enable_perlayer_output []
                        enable dump perlayer output. type: bool. required: false. default: false.
  --dump_with_stride []
  --group_index
  --mode                run mode, only work for QuantAxModel. type: enum. required: false. default: Reference.
                        option: Reference, NPUBackend.
  --target_hardware     target hardware, only work for QuantAxModel. type: enum. required: false. default: AX650.
                        option: AX650, AX620E, M76H.

optionalパラメータ:
- `--model`: 実行するモデルのパス（ONNX、QuantAxModel、CompiledAxmodelをサポート）
- `--input_dir`: モデルの入力データのディレクトリ
- `--output_dir`: モデル出力データの保存先ディレクトリ
- `--list`: リストファイルのパス

オプションパラメータ:
- `--config`: 設定ファイルのパス（json/yaml/toml/prototxtをサポート）
- `--random_input`: ランダムな入力データを使用（デフォルト: false）
- `--batch_size`: 動的推論モードで使用するバッチサイズ（CompiledAxModelのみ有効）
- `--enable_perlayer_output`: レイヤーごとの出力ダンプを有効化（デフォルト: false）
- `--mode`: 実行モード（QuantAxModelのみ有効）
  - オプション: Reference、NPUBackend
  - デフォルト: Reference
- `--target_hardware`: ターゲットハードウェア（QuantAxModelのみ有効）
  - オプション: AX650、AX620E、M76H
  - デフォルト: AX650

```



