import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import onnx
import argparse


def get_input_info(model_path: str) -> List[str]:
    """
    Returns the names list of axmodel.
    """

    model_obj = onnx.load(model_path)

    if hasattr(model_obj, "graph"):
        inits = set(_.name for _ in model_obj.graph.initializer)
        return [_.name for _ in model_obj.graph.input if _.name not in inits]
    return list(model_obj.input)


def preprocess(img: np.ndarray, input_width: int = 640, input_height: int = 480) -> np.ndarray:
    # より効率的な色変換を使用
    if len(img.shape) == 3 and img.shape[2] == 3:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = img

    # 高速なリサイズにINTER_LINEARを使用
    resized = cv2.resize(image_rgb, (input_width, input_height),
                       interpolation=cv2.INTER_LINEAR)

  # 正規化とトランスポーズを1ステップで実行
    input_tensor = resized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0
  

    #input_tensor = np.expand_dims(resized, axis=0)
    print(f"入力テンソル形状: {input_tensor.shape}")
    return input_tensor


    

def save_preprocessed_data(
    image_path: str,
    axmodel_path: str, 
    intermediate_path: str,
    input_width: int = 640,
    input_height: int = 480
):
    # 入力ファイルの存在チェック
    if not Path(image_path).exists():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
    if not Path(axmodel_path).exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {axmodel_path}")
    if not Path(intermediate_path).exists():
        raise FileNotFoundError(f"中間ファイル出力先が見つかりません: {intermediate_path}")

    # 画像読み込み
    img = cv2.imread(image_path)
    
    # 前処理実行
    img_transformed = preprocess(img, input_width, input_height)

    # モデルの入力名を取得
    input_names = get_input_info(axmodel_path)
    if len(input_names) != 1:
        raise NotImplementedError(f"入力は1つのみサポートしています。現在: {input_names}")

    # 形状情報をコンソールに出力
    print(f"データ型: {img_transformed.dtype}")
    
    base_path = Path(intermediate_path) / input_names[0]

    # バイナリファイルとして保存
    bin_path = base_path.with_suffix('.bin')
    bin_path.write_bytes(img_transformed.tobytes())
    print(f"中間ファイル(バイナリ)を保存しました: {bin_path}")

    # numpy形式で保存
    npy_path = base_path.with_suffix('.npy')
    np.save(npy_path, img_transformed)
    print(f"中間ファイル(numpy)を保存しました: {npy_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='画像の前処理を行い中間ファイルを生成します')
    parser.add_argument('--image', required=False, default='input.jpg', help='入力画像のパス')
    parser.add_argument('--model', required=False, default='depth_anything_u8.axmodel', help='モデルファイルのパス')
    parser.add_argument('--output', required=False, default='sim_inputs/0', help='中間ファイルの出力先ディレクトリ')
    parser.add_argument('--width', type=int, default=384, help='リサイズ後の幅 (デフォルト: 384)')
    parser.add_argument('--height', type=int, default=256, help='リサイズ後の高さ (デフォルト: 256)')
    
    args = parser.parse_args()
    
    save_preprocessed_data(
        image_path=args.image,
        axmodel_path=args.model,
        intermediate_path=args.output,
        input_width=args.width,
        input_height=args.height
    )


