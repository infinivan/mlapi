#python debian
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

#set working dir
WORKDIR /app

#copy local files to container
COPY . /app

#dependencies
RUN pip install -r requirements.txt


#starting command
#CMD ["uvicorn","main:app","--reload","--port","5000"]
#CMD ["python","main.py"]