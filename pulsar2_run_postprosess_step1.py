import numpy as np
from pathlib import Path
import onnx
import argparse
import cv2

def get_tensor_value_info(tensor_info: onnx.TensorProto):
    name = tensor_info.name

    shape = None
    elem_type = None
    if tensor_info.HasField("type"):
        shape = []
        tensor_type = tensor_info.type.tensor_type
        elem_type = tensor_type.elem_type
        for d in tensor_type.shape.dim:
            # the dimension may have a definite (integer) value or a symbolic identifier or neither:
            if d.HasField("dim_value"):  # known dimension
                shape.append(int(d.dim_value))
            elif d.HasField("dim_param"):  # unknown dimension with symbolic name
                shape.append(str(d.dim_param))
            else:  # unknown dimension with no name
                shape.append(-1)
    return {"name": name, "shape": shape, "tensor_type": _elem_type_as_numpy(elem_type)}


def _elem_type_as_numpy(elem_type):
    if elem_type == onnx.TensorProto.FLOAT:
        return np.dtype("float32")
    if elem_type == onnx.TensorProto.INT32:
        return np.dtype("int32")
    if elem_type == onnx.TensorProto.UINT32:
        return np.dtype("uint32")
    if elem_type == onnx.TensorProto.UINT64:
        return np.dtype("uint64")
    if elem_type == onnx.TensorProto.INT16:
        return np.dtype("int16")
    if elem_type == onnx.TensorProto.UINT16:
        return np.dtype("uint16")
    if elem_type == onnx.TensorProto.UINT8:
        return np.dtype("uint8")
    if elem_type == onnx.TensorProto.INT8:
        return np.dtype("int8")
    if elem_type == onnx.TensorProto.FLOAT16:
        return np.dtype("float16")
    raise NotImplementedError(f"Currently doesn't support type: '{elem_type}'.")


def get_output_info(model_path: str):
    """
    Returns the shape and tensor type of all outputs.
    """
    model_obj = onnx.load(model_path)
    model_graph = model_obj.graph

    output_info = {}
    for tensor_info in model_graph.output:
        output_info.update({tensor_info.name: get_tensor_value_info(tensor_info)})


    output_file= "model_info.txt"

    # テキストファイルとして保存
    with open(output_file, 'w') as f:
        f.write("ONNXモデル情報:\n")
        f.write(f"モデルパス: {model_path}\n\n")
        
        f.write("出力テンソル情報:\n")
        for name, info in output_info.items():
            f.write(f"\n出力名: {name}\n")
            f.write(f"形状: {info['shape']}\n")
            f.write(f"データ型: {info['tensor_type']}\n")

    # モデルを保存
    onnx.save(model_obj, "test.onnx")

    return output_info

def save_as_image(array: np.ndarray, output_dir: str, filename: str):
    """
    numpy配列をOpenCV画像として保存します。
    
    Args:
        array: 入力numpy配列
        output_dir: 出力ディレクトリ
        filename: 出力ファイル名
    """
    output_dir = Path(output_dir)
    
    # (1, 1, 256, 384)の形状から(256, 384)の形状に変換
    img_array = array[0, 0]  # バッチとチャンネル次元を削除
    
    # 255倍してuint8に変換
    img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
    
    img_path = output_dir / f"{filename}.png"
    cv2.imwrite(str(img_path), img_array)
    print(f"画像ファイルを保存しました: {img_path}")



def convert_bin_to_npy(
    bin_path: str,
    shape: tuple,
    tensor_type: np.dtype,
    output_dir: str = "./"
):
    """
    binファイルをnpy形式に変換します。
    
    Args:
        bin_path: 入力binファイルのパス
        shape: 出力テンソルの形状 (例: (1, 89, 60, 80))
        tensor_type: 出力テンソルのデータ型 (例: np.float32)
        output_dir: 出力ディレクトリ
    """
    # パスの処理
    bin_path = Path(bin_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 入力ファイルの存在確認
    if not bin_path.exists():
        raise FileNotFoundError(f"binファイルが見つかりません: {bin_path}")

    # binファイルを読み込み
    binary_data = bin_path.read_bytes()
    
    # numpy配列に変換
    array = np.frombuffer(binary_data, dtype=tensor_type)
    array = array.reshape(shape)
    

    # npy形式で保存
    output_path = output_dir / f"{bin_path.stem}.npy"
    np.save(output_path, array)
    print(f"npyファイルを保存しました: {output_path}")
    # 情報を表示
    print(f"データ形状: {array.shape}")
    print(f"データ型: {array.dtype}")
    

    # OpenCV画像として保存
    save_as_image(array, output_dir, bin_path.stem)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='binファイルをnpy形式に変換し、モデル情報を取得します')
    parser.add_argument('--model', required=True, help='モデルファイルのパス')
    parser.add_argument('--output-dir', required=True, help='出力ディレクトリ')
    parser.add_argument('--num-outputs', type=int, required=True, help='出力の数')
    
    args, remaining_args = parser.parse_known_args()
    
    # 動的に出力の数に応じた引数を追加
    for i in range(args.num_outputs):
        parser.add_argument(f'--bin{i+1}', required=True, help=f'{i+1}番目のbinファイルのパス')
        parser.add_argument(f'--shape{i+1}', nargs='+', type=int, help=f'{i+1}番目の出力形状')
    
    args = parser.parse_args()
    
    # モデルから出力情報を取得
    output_info = get_output_info(args.model)
    print("モデルの出力情報:")
    for k, v in output_info.items():
        print(f"  出力名: {k}")
        print(f"  形状: {v['shape']}")
        print(f"  データ型: {v['tensor_type']}")
        print()

    # 動的に各出力に対して変換を実行
    for i in range(args.num_outputs):
        bin_path = getattr(args, f'bin{i+1}')
        shape = getattr(args, f'shape{i+1}')
        
        # shapeが指定されていない場合はoutput_infoから取得
        if not shape and len(output_info) > i:
            shape = list(output_info.values())[i]['shape']
        
        if shape:
            convert_bin_to_npy(
                bin_path=bin_path,
                shape=tuple(shape),
                tensor_type=np.float32,
                output_dir=args.output_dir
            )


"""
# 3つの出力の場合
python script.py \
  --model depth_anything_u8.axmodel \
  --output-dir ./sim_outputs/0 \
  --num-outputs 3 \
  --bin1 ./sim_outputs/0/output1.bin \
  --shape1 1 89 60 80 \
  --bin2 ./sim_outputs/0/output2.bin \
  --shape2 1 89 30 40 \
  --bin3 ./sim_outputs/0/output3.bin \
  --shape3 1 89 15 20

# 2つの出力の場合
python pulsar2_run_postprosess_step1.py \
  --model depth_anything_u8.axmodel \
  --output-dir ./sim_outputs/0 \
  --num-outputs 1 \
  --bin1 ./sim_outputs/0/depth.bin 
"""