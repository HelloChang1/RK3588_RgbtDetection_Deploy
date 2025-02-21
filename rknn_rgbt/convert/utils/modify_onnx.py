import onnx
import onnx_graphsurgeon as gs
import argparse
import onnxsim
from onnx.shape_inference import infer_shapes
import os

# 这个函数接受一个 ONNX 计算图对象 graph。
def modify_onnx(graph):
    
    # 获取图中所有节点
    nodes = graph.nodes
    #  初始化空字典，将节点存储在字典中，以节点名称为键
    nodes_dict = {}
    # 初始化一个列表，用于存储输出节点名称
    outputs_nodes = []
    # 遍历图中的所有节点，将节点的名称作为键，节点对象作为值存入 nodes_dict 字典中
    for node in nodes:
        name = node.name
        nodes_dict.update({name: node})

    for k, v in nodes_dict.items():
        if v.op == "Reshape":
            if v.inputs[0].inputs[0].op == "Conv":
                # 获取卷积层的名称 conv_name
                conv_name = v.inputs[0].inputs[0].name
                # 获取 Reshape 操作的输出形状 shape
                if len(v.outputs) > 0 and v.outputs[0] is not None:
                    shape = v.outputs[0].shape
                    print(f"Reshape shape: {shape}")
                else:
                    print(f"Warning: Reshape node '{v.name}' has no valid outputs.")
                if conv_name in nodes_dict:
                    conv_node = nodes_dict[conv_name]
                    if conv_node.outputs is not None and len(conv_node.outputs) > 0:
                        print(f"Before modification: {conv_node.outputs[0].shape}")                     
                # 修改对应卷积层输出的形状，形状变为 (1, shape[1]*shape[2], shape[3], shape[4])，这种改动可能是为了优化后续数据处理
                        conv_node.outputs[0].shape = (1, shape[1]*shape[2], shape[3], shape[4])
                    else:
                        print(f"Outputs are not available for {conv_name}")
                else:
                    print(f"{conv_name} not found in nodes_dict")

                outputs_nodes.append(conv_name)
    # 调整输出: 确保图的输出数量不超过 3，不够时用第一个输出填充，超过时则裁剪。
    while len(graph.outputs) <= 3:
        graph.outputs.append(graph.outputs[0])
    if len(graph.outputs) > 3:
        graph.outputs = graph.outputs[:3]

    # 更新输出节点: 根据 outputs_nodes 中的记录，更新计算图的输出。
    for idx, val in enumerate(outputs_nodes):
        graph.outputs[idx] = nodes_dict[val].outputs[0]

    # 使用 cleanup() 方法移除任何不再使用的节点，并使用 toposort() 进行拓扑排序以保持图的结构。
    graph.cleanup().toposort()
    print("After modify Nodes:{}".format(len(graph.nodes)))
    return graph

# 加载模型函数
def load_model(path):
    print("Load onnx model from {}".format(path))
    onnx_model = infer_shapes(onnx.load(path))
    graph = gs.import_onnx(onnx_model)
    print("Before modify Nodes:{}".format(len(graph.nodes)))
    graph.fold_constants().cleanup()
    return graph

# 保存模型函数
def save_model(graph, path):
    print('Starting to simplify ONNX...')
    onnx_model, check = onnxsim.simplify(gs.export_onnx(graph))
    onnx.save(onnx_model, path)
    print("Save modified onnx model to {}".format(path))


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('-i', '--input', required=True,
                        type=str, help='Input onnx model')
    parser.add_argument('-o', '--output', type=str,
                        default='model/onnx_modify/', help='Save modeified onnx model path')

    opt = parser.parse_args()

    model_path = opt.input
    save_path = os.path.join(opt.output, model_path.split("/")[-1])
    graph = load_model(model_path)
    graph = modify_onnx(graph)
    save_model(graph, save_path)
