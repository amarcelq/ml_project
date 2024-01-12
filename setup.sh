python3.9 -m venv venv_3_9
#echo "Now activate venv with:"
source ./venv_3_9/bin/activate
#echo "Then install requirements with:"
pip3.9 install -r requirements
echo "Start Jupyter Lab with:"
python3.9 -m jupyterlab .
