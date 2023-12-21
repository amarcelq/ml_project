python3.11 -m venv venv
#echo "Now activate venv with:"
source ./venv/bin/activate
#echo "Then install requirements with:"
pip3.11 install -r requirements
echo "Start Jupyter Lab with:"
python3.11 -m jupyterlab .
