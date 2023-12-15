# check for aguments $1

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

if [ $# -eq 0 ]
then
  echo "No arguments supplied\nUsage: $0 jupyter-notebook.ipynb [output-name]\n"
  exit
fi

FILE_NAME=$1
if [ -z "$2" ]
then
  OUTPUT_NAME=$(echo $1 | sed 's/ipynb/py/')
else
  OUTPUT_NAME=$2
fi

jupyter nbconvert --to script $FILE_NAME --output  $OUTPUT_NAME

grep -v '^get_ipython' $OUTPUT_NAME.py > tempfile && mv tempfile $OUTPUT_NAME.py
nice -n -20 sh -c "python3 $OUTPUT_NAME.py &" & pid=$!

wait $pid 
echo "Process has ID $pid"
exit
