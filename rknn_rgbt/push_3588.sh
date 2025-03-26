echo "start push to rk3588:{ip}:{port},please wait"
scp  -P {port} -r ./install/rknn_detection_model_Linux root@{ip}:{path}
