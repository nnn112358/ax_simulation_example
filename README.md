
## Purpose

Simulate a depth anything using the ax630c simulator pulsar2 run.


# pulsar2 document 

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

optional�p�����[�^:
- `--model`: ���s���郂�f���̃p�X�iONNX�AQuantAxModel�ACompiledAxmodel���T�|�[�g�j
- `--input_dir`: ���f���̓��̓f�[�^�̃f�B���N�g��
- `--output_dir`: ���f���o�̓f�[�^�̕ۑ���f�B���N�g��
- `--list`: ���X�g�t�@�C���̃p�X

�I�v�V�����p�����[�^:
- `--config`: �ݒ�t�@�C���̃p�X�ijson/yaml/toml/prototxt���T�|�[�g�j
- `--random_input`: �����_���ȓ��̓f�[�^���g�p�i�f�t�H���g: false�j
- `--batch_size`: ���I���_���[�h�Ŏg�p����o�b�`�T�C�Y�iCompiledAxModel�̂ݗL���j
- `--enable_perlayer_output`: ���C���[���Ƃ̏o�̓_���v��L�����i�f�t�H���g: false�j
- `--mode`: ���s���[�h�iQuantAxModel�̂ݗL���j
  - �I�v�V����: Reference�ANPUBackend
  - �f�t�H���g: Reference
- `--target_hardware`: �^�[�Q�b�g�n�[�h�E�F�A�iQuantAxModel�̂ݗL���j
  - �I�v�V����: AX650�AAX620E�AM76H
  - �f�t�H���g: AX650

```


## How to

Enter Docker in pulsar2

```
 sudo docker run -it --net host --rm -v $PWD:/data pulsar2:3.3
```

Check the interface of depth_anything_u8.axmodel

```
# python axmodel_get_info.py depth_anything_u8.axmodel
INFO: ���f����ǂݍ��ݒ�: depth_anything_u8.axmodel

=== ONNX���f����� ===
���f���p�X: depth_anything_u8.axmodel
IR�o�[�W����: 8
�v���f���[�T�[: Pulsar2 (�o�[�W����: )
�h���C��:
���f���o�[�W����: 0

=== ���̓e���\����� ===

���͖�: image
�`��: [1, 256, 384, 3]
�f�[�^�^: uint8
���v�f��: 294,912

=== �o�̓e���\����� ===

�o�͖�: depth
�`��: [1, 1, 256, 384]
�f�[�^�^: float32
���v�f��: 98,304
```



```
# python pulsar2_run_preprocess_axsim.py
# pulsar2 run --model depth_anything_u8.axmodel --input_dir sim_inputs --output_dir sim_outputs --list list.txt
# python pulsar2_run_postprosess_step1.py   --model depth_anything_u8.axmodel   --output-dir ./sim_outputs/0   --num-outputs 1   --bin1 ./sim_outputs/0/depth.bin

```


```
# python pulsar2_run_preprocess_onnx.py
# pulsar2 run --model depth_anything_u8.axmodel --input_dir sim_inputs --output_dir sim_outputs --list list.txt
# python pulsar2_run_postprosess_step1.py   --model depth_anything_u8.axmodel   --output-dir ./sim_outputs/0   --num-outputs 1   --bin1 ./sim_outputs/0/depth.bin

```

```
```


