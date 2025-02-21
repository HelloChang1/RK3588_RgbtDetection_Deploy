echo "start push to rk3588:619.grifcc.top:60001,please wait"
scp  -P 60001 -r ./install/rknn_detection_model_Linux root@619.grifcc.top:/workspace/detect/
