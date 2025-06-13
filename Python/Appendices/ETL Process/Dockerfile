FROM python:3.9
RUN apt update && apt install libpq-dev python3-dev gcc -y 
RUN python -m pip install --upgrade pip && pip3 install notebook==6.4.13 ipython-sql==0.4.1 psycopg2==2.9.4  
RUN pip3 uninstall -y sqlalchemy
RUN pip3 install sqlalchemy==1.4.41  
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.token = ''" >> $HOME/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = u''" >> $HOME/.jupyter/jupyter_notebook_config.py
CMD jupyter notebook --allow-root  --ip 0.0.0.0 --no-browser --port 8888
