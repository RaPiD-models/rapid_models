FROM python:3.8-alpine

RUN apk update
RUN apk add --no-cache python3 py3-pip

# Install python packages for sphinx build
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r requirements_dev.txt

#WORKDIR /docs
# Copy in docs folder for building
#COPY ./package.json .
#COPY ./yarn.lock .
#RUN yarn install

ENV NODE_ENV=production

COPY docs docs
COPY src src

# Build docs to docs/_build
#RUN cd docs
RUN ./docs/sphinx-apidoc -o . ../src/rapid_models
RUN ./docs/make html

# Build package
RUN python setup.py
RUN mkdir ./docs/build/html/dist
RUN mv ./dist/*.tar.gz ./docs/build/html/dist/

# Create new image
FROM nginx:alpine

WORKDIR /app
# Copy the static build assets to /app dir
COPY --from=0 ./docs/build/html/ .
# Copy in the nginx config file
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
# All files are in, start the web server
CMD ["nginx"]