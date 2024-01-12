python3.10 -m venv venv_3_10
#echo "Now activate venv with:"
source ./venv_3_10/bin/activate
#echo "Then install requirements with:"
pip3.10 install -r requirements
echo "Start Jupyter Lab with:"
python3.10 -m jupyterlab .
