conda activate orphicx
loss function f gae optimizer

### Retrain OrphicX from scratch

python orphicx_lyax.py --dataset syn1 --output syn1_retrain --retrain
python orphicx_node.py --dataset syn4 --output syn4_retrain --retrain
python orphicx_graph.py --dataset Mutagenicity --output mutag_retrain --retrain
python orphicx_graph.py --dataset NCI1 --output nci1_retrain --retrain

### eval

python orphicx_node.py --dataset syn1 --plot_info_flow
python orphicx_lyas_brain.py --dataset syn1 --plot_info_flow
python orphicx_node.py --dataset syn4 --plot_info_flow
python orphicx_graph.py --dataset Mutagenicity --plot_info_flow
python orphicx_graph.py --dataset NCI1 --plot_info_flow

### lyax

python orphicx_graph.py --dataset NCI1 --plot_info_flow --gcn_mp_type edge_node_concate --retrain
python orphicx_graph_brain.py --dataset NCI1 --plot_info_flow --gcn_mp_type edge_node_concate --batch_size 1 --retrain
