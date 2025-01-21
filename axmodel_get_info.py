import numpy as np
from pathlib import Path
import onnx
import argparse
import sys
from typing import Dict, Any, Tuple
import logging

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='ONNXモデルの情報を解析して表示します')
    parser.add_argument('model_path', type=str, help='ONNXモデルファイルのパス')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='詳細な情報を表示します')
    return parser

def get_tensor_value_info(tensor_info: onnx.TensorProto) -> Dict[str, Any]:
    try:
        name = tensor_info.name
        shape = None
        elem_type = None
        if tensor_info.HasField("type"):
            shape = []
            tensor_type = tensor_info.type.tensor_type
            elem_type = tensor_type.elem_type
            for d in tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    shape.append(int(d.dim_value))
                elif d.HasField("dim_param"):
                    shape.append(str(d.dim_param))
                else:
                    shape.append(-1)
        
        return {
            "name": name,
            "shape": shape,
            "tensor_type": elem_type_as_numpy(elem_type),
            "total_elements": np.prod([d for d in shape if isinstance(d, int) and d > 0]) if shape else None
        }
    except Exception as e:
        logger.error(f"テンソル情報の解析中にエラーが発生: {e}")
        raise

def elem_type_as_numpy(elem_type: int) -> np.dtype:
    type_mapping = {
        onnx.TensorProto.FLOAT: np.dtype("float32"),
        onnx.TensorProto.INT32: np.dtype("int32"),
        onnx.TensorProto.UINT32: np.dtype("uint32"),
        onnx.TensorProto.UINT64: np.dtype("uint64"),
        onnx.TensorProto.INT16: np.dtype("int16"),
        onnx.TensorProto.UINT16: np.dtype("uint16"),
        onnx.TensorProto.UINT8: np.dtype("uint8"),
        onnx.TensorProto.INT8: np.dtype("int8"),
        onnx.TensorProto.FLOAT16: np.dtype("float16"),
        onnx.TensorProto.BOOL: np.dtype("bool"),
        onnx.TensorProto.DOUBLE: np.dtype("float64"),
        onnx.TensorProto.INT64: np.dtype("int64"),
    }
    
    if elem_type not in type_mapping:
        raise NotImplementedError(f"未サポートのデータ型です: '{elem_type}'")
    return type_mapping[elem_type]

def get_model_info(model_path: str, verbose: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

        logger.info(f"モデルを読み込み中: {model_path}")
        model_obj = onnx.load(str(model_path))
        model_graph = model_obj.graph

        model_info = {
            "ir_version": model_obj.ir_version,
            "producer_name": model_obj.producer_name,
            "producer_version": model_obj.producer_version,
            "domain": model_obj.domain,
            "model_version": model_obj.model_version,
            "doc_string": model_obj.doc_string,
        }

        input_info = {tensor_info.name: get_tensor_value_info(tensor_info) 
                     for tensor_info in model_graph.input}
        output_info = {tensor_info.name: get_tensor_value_info(tensor_info) 
                      for tensor_info in model_graph.output}

        if verbose:
            nodes_info = []
            for node in model_graph.node:
                nodes_info.append({
                    "op_type": node.op_type,
                    "input": list(node.input),
                    "output": list(node.output)
                })
            model_info["nodes"] = nodes_info
            model_info["initializer_count"] = len(model_graph.initializer)

        return input_info, output_info, model_info

    except Exception as e:
        logger.error(f"モデル解析中にエラーが発生: {e}")
        raise

def display_model_info(model_path: str, input_info: Dict[str, Any], 
                      output_info: Dict[str, Any], model_info: Dict[str, Any],
                      verbose: bool = False):
    """モデル情報をコンソールに表示する"""
    print("\n=== ONNXモデル情報 ===")
    print(f"モデルパス: {model_path}")
    print(f"IRバージョン: {model_info['ir_version']}")
    print(f"プロデューサー: {model_info['producer_name']} "
          f"(バージョン: {model_info['producer_version']})")
    print(f"ドメイン: {model_info['domain']}")
    print(f"モデルバージョン: {model_info['model_version']}")
    
    if verbose and model_info.get("nodes"):
        print(f"\n総ノード数: {len(model_info['nodes'])}")
        print(f"初期化子数: {model_info['initializer_count']}")
        
        # オペレーション種類の集計
        op_types = {}
        for node in model_info['nodes']:
            op_types[node['op_type']] = op_types.get(node['op_type'], 0) + 1
        
        print("\nオペレーション種類の内訳:")
        for op_type, count in sorted(op_types.items()):
            print(f"  - {op_type}: {count}個")

    print("\n=== 入力テンソル情報 ===")
    for name, info in input_info.items():
        print(f"\n入力名: {name}")
        print(f"形状: {info['shape']}")
        print(f"データ型: {info['tensor_type']}")
        if info['total_elements']:
            print(f"総要素数: {info['total_elements']:,}")

    print("\n=== 出力テンソル情報 ===")
    for name, info in output_info.items():
        print(f"\n出力名: {name}")
        print(f"形状: {info['shape']}")
        print(f"データ型: {info['tensor_type']}")
        if info['total_elements']:
            print(f"総要素数: {info['total_elements']:,}")

def main():
    parser = setup_argument_parser()
    args = parser.parse_args()

    try:
        input_info, output_info, model_info = get_model_info(args.model_path, args.verbose)
        display_model_info(args.model_path, input_info, output_info, model_info, args.verbose)

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()