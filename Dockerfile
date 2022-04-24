FROM python:3.8.5

# run this before copying requirements for cache efficiency
RUN pip install --upgrade pip

#set work directory early so remaining paths can be relative
WORKDIR /

# Adding requirements file to current directory
# just this file first to cache the pip install step when code changes
COPY requirements.txt .

# copy code itself from context to image
COPY . .

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

CMD [ "python", "api.py"]